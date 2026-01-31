"""
Threshold-Based Elastic Routing for EDP.

Key features:
- No top-k competition: each token decides independently
- Learned per-layer thresholds
- Bowl-shaped routing: always-on early/late layers, gated middle layers
- FLOP-aware relative depth encoding for warp-forward

This replaces MoD-style competitive routing with truly elastic depth allocation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import math

from ..utils.flops import layer_flops, flops_to_bucket


class LearnedThreshold(nn.Module):
    """
    Learned per-layer threshold for routing decisions.
    
    gate = (signal > tau).float()
    
    tau is learned via straight-through estimator.
    """
    
    def __init__(
        self,
        n_layers: int,
        initial_value: float = 0.1,
        min_value: float = 0.01,
        max_value: float = 1.0,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.min_value = min_value
        self.max_value = max_value
        
        # Learnable threshold per layer (in logit space for stable optimization)
        # Initialize to achieve initial_value after sigmoid
        init_logit = math.log(initial_value / (1 - initial_value + 1e-8))
        self.tau_logits = nn.Parameter(torch.full((n_layers,), init_logit))
        
    def forward(self, layer_idx: int) -> torch.Tensor:
        """Get threshold for a specific layer."""
        tau = torch.sigmoid(self.tau_logits[layer_idx])
        # Scale to [min_value, max_value]
        tau = self.min_value + tau * (self.max_value - self.min_value)
        return tau
    
    def get_all_thresholds(self) -> torch.Tensor:
        """Get all thresholds."""
        tau = torch.sigmoid(self.tau_logits)
        return self.min_value + tau * (self.max_value - self.min_value)


class FixedThreshold(nn.Module):
    """
    Fixed threshold for ablation comparison.
    """
    
    def __init__(self, n_layers: int, value: float = 0.1):
        super().__init__()
        self.register_buffer('tau', torch.full((n_layers,), value))
        
    def forward(self, layer_idx: int) -> torch.Tensor:
        return self.tau[layer_idx]
    
    def get_all_thresholds(self) -> torch.Tensor:
        return self.tau


class ThresholdRouter(nn.Module):
    """
    Threshold-based routing with straight-through gradient estimation.
    
    Computes: gate = (signal > tau).float()
    
    Uses straight-through estimator for gradients through the discrete decision.
    """
    
    def __init__(
        self,
        n_layers: int,
        learned: bool = True,
        initial_threshold: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.learned = learned
        self.temperature = temperature
        
        if learned:
            self.threshold = LearnedThreshold(n_layers, initial_value=initial_threshold)
        else:
            self.threshold = FixedThreshold(n_layers, value=initial_threshold)
            
    def forward(
        self,
        signal: torch.Tensor,
        layer_idx: int,
        hard: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute routing decision.
        
        Args:
            signal: Routing signal [batch, seq_len]
            layer_idx: Current layer index
            hard: If True, use hard thresholding with straight-through gradient
            
        Returns:
            gate: Binary gate [batch, seq_len]
            soft_gate: Soft gate for gradient flow [batch, seq_len]
        """
        tau = self.threshold(layer_idx)
        
        # Soft gate (for gradient flow)
        # Higher signal → more likely to compute → gate closer to 1
        soft_gate = torch.sigmoid((signal - tau) / self.temperature)
        
        if hard:
            # Hard threshold with straight-through estimator
            hard_gate = (signal > tau).float()
            # Straight-through: forward uses hard, backward uses soft
            gate = hard_gate - soft_gate.detach() + soft_gate
        else:
            gate = soft_gate
            
        return gate, soft_gate
    
    def get_thresholds(self) -> torch.Tensor:
        """Get all threshold values for logging."""
        return self.threshold.get_all_thresholds()


class FLOPAwareStepEncoding(nn.Module):
    """
    FLOP-aware relative depth encoding.
    
    Instead of: x += step_embedding[layer_id]
    We use:     x += step_embedding[cumulative_skipped_flops_bucket]
    
    This encodes semantic age (how much compute was skipped), not absolute depth.
    Critical for warp-forward to work correctly.
    """
    
    def __init__(
        self,
        d_model: int,
        num_buckets: int = 64,
        max_skip_layers: int = 12,
        seq_len: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_buckets = num_buckets
        
        # Compute max possible skipped FLOPs
        self.flops_per_layer = layer_flops(d_model, seq_len)
        self.max_skipped_flops = max_skip_layers * self.flops_per_layer
        
        # Learnable step embeddings indexed by FLOP bucket
        self.step_embeddings = nn.Embedding(num_buckets, d_model)
        
        # Initialize small
        nn.init.normal_(self.step_embeddings.weight, std=0.02)
        
    def forward(
        self,
        x: torch.Tensor,
        cumulative_skipped_flops: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add FLOP-aware step encoding to hidden states.
        
        Args:
            x: Hidden states [batch, seq_len, d_model]
            cumulative_skipped_flops: FLOPs skipped so far per token [batch, seq_len]
            
        Returns:
            x: Hidden states with step encoding added
        """
        # Convert FLOPs to bucket index
        bucket_idx = flops_to_bucket(
            cumulative_skipped_flops,
            self.max_skipped_flops,
            self.num_buckets,
        )
        
        # Get step embeddings
        step_emb = self.step_embeddings(bucket_idx)
        
        return x + step_emb


class StaticStepEncoding(nn.Module):
    """
    Static (layer-based) step encoding for ablation comparison.
    
    x += step_embedding[layer_id]
    """
    
    def __init__(self, d_model: int, n_layers: int):
        super().__init__()
        self.step_embeddings = nn.Embedding(n_layers, d_model)
        nn.init.normal_(self.step_embeddings.weight, std=0.02)
        
    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Add static step encoding.
        
        Args:
            x: Hidden states [batch, seq_len, d_model]
            layer_idx: Current layer index
            
        Returns:
            x: Hidden states with step encoding added
        """
        step_emb = self.step_embeddings.weight[layer_idx]
        return x + step_emb.unsqueeze(0).unsqueeze(0)


class ElasticRouter(nn.Module):
    """
    Complete elastic routing module for EDP.
    
    Combines:
    - Threshold-based routing
    - Bowl-shaped layer selection
    - FLOP-aware step encoding
    - Warp-forward logic
    """
    
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        early_layers: int = 3,
        late_layers: int = 3,
        learned_threshold: bool = True,
        initial_threshold: float = 0.1,
        use_flop_aware_encoding: bool = True,
        num_step_buckets: int = 64,
        max_skip_layers: int = 4,
        seq_len: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.early_layers = early_layers
        self.late_layers = late_layers
        self.max_skip_layers = max_skip_layers
        
        # Number of gatable (middle) layers
        self.n_middle_layers = n_layers - early_layers - late_layers
        assert self.n_middle_layers > 0, "Must have at least one gatable layer"
        
        # Middle layer indices
        self.middle_layer_start = early_layers
        self.middle_layer_end = n_layers - late_layers
        
        # Router
        self.router = ThresholdRouter(
            n_layers=self.n_middle_layers,
            learned=learned_threshold,
            initial_threshold=initial_threshold,
        )
        
        # Step encoding
        if use_flop_aware_encoding:
            self.step_encoding = FLOPAwareStepEncoding(
                d_model=d_model,
                num_buckets=num_step_buckets,
                max_skip_layers=n_layers,
                seq_len=seq_len,
            )
        else:
            self.step_encoding = StaticStepEncoding(d_model, n_layers)
            
        self.use_flop_aware_encoding = use_flop_aware_encoding
        self.flops_per_layer = layer_flops(d_model, seq_len)
        
    def is_always_on(self, layer_idx: int) -> bool:
        """Check if layer is always-on (early or late)."""
        return layer_idx < self.early_layers or layer_idx >= (self.n_layers - self.late_layers)
    
    def get_middle_layer_idx(self, layer_idx: int) -> int:
        """Convert global layer index to middle layer index."""
        return layer_idx - self.early_layers
    
    def forward(
        self,
        signal: torch.Tensor,
        layer_idx: int,
        cumulative_skipped_flops: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute routing decision for a layer.
        
        Args:
            signal: Routing signal [batch, seq_len]
            layer_idx: Global layer index
            cumulative_skipped_flops: FLOPs skipped so far [batch, seq_len]
            
        Returns:
            gate: Binary gate [batch, seq_len]
            soft_gate: Soft gate for gradient [batch, seq_len]
            updated_skipped_flops: Updated cumulative skipped FLOPs
        """
        batch_size, seq_len = signal.shape
        device = signal.device
        
        if self.is_always_on(layer_idx):
            # Always-on layer: gate = 1
            gate = torch.ones(batch_size, seq_len, device=device)
            soft_gate = gate
        else:
            # Gatable layer: use router
            middle_idx = self.get_middle_layer_idx(layer_idx)
            gate, soft_gate = self.router(signal, middle_idx)
            
        # Update cumulative skipped FLOPs
        skipped = (1 - gate) * self.flops_per_layer
        updated_skipped_flops = cumulative_skipped_flops + skipped
        
        return gate, soft_gate, updated_skipped_flops
    
    def apply_step_encoding(
        self,
        x: torch.Tensor,
        layer_idx: int,
        cumulative_skipped_flops: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply step encoding to hidden states.
        
        Args:
            x: Hidden states [batch, seq_len, d_model]
            layer_idx: Current layer index
            cumulative_skipped_flops: FLOPs skipped so far [batch, seq_len]
            
        Returns:
            x: Hidden states with step encoding
        """
        if self.use_flop_aware_encoding:
            return self.step_encoding(x, cumulative_skipped_flops)
        else:
            return self.step_encoding(x, layer_idx)
    
    def get_routing_stats(self) -> Dict[str, torch.Tensor]:
        """Get routing statistics for logging."""
        return {
            'thresholds': self.router.get_thresholds(),
            'early_layers': self.early_layers,
            'late_layers': self.late_layers,
            'n_middle_layers': self.n_middle_layers,
        }
