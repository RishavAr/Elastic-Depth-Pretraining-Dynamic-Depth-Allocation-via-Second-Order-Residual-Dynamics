"""
FLOPs Accounting for Elastic-Depth Pretraining.

This module provides exact FLOPs calculations used throughout:
- Routing decisions
- Relative depth encoding
- Plots and paper claims

Per token per layer, dense FLOPs:
| Component         | FLOPs           |
|-------------------|-----------------|
| QKV projections   | 3 * d²          |
| Output projection | 1 * d²          |
| Attention softmax | 2 * seq_len * d |
| FFN up + down     | 8 * d²          |
| Total             | 12 * d² + 2 * seq_len * d |
"""
import torch
from typing import Optional, Dict


def layer_flops(d_model: int, seq_len: int) -> int:
    """
    Compute FLOPs for a single layer per token.
    
    Args:
        d_model: Model dimension
        seq_len: Sequence length
        
    Returns:
        Number of FLOPs per token for one layer
    """
    # QKV projections: 3 * d²
    qkv_flops = 3 * d_model ** 2
    
    # Output projection: d²
    output_proj_flops = d_model ** 2
    
    # Attention softmax: 2 * seq_len * d (QK^T and weighted sum)
    attention_flops = 2 * seq_len * d_model
    
    # FFN: 2 * (d * d_ffn) = 2 * d * 4d = 8d² (up and down projections)
    ffn_flops = 8 * d_model ** 2
    
    return qkv_flops + output_proj_flops + attention_flops + ffn_flops


def token_flops(gate: torch.Tensor, d_model: int, seq_len: int) -> torch.Tensor:
    """
    Compute FLOPs used by tokens given their gate values.
    
    Args:
        gate: Binary gate tensor [batch, seq_len] or [batch, seq_len, n_layers]
        d_model: Model dimension
        seq_len: Sequence length
        
    Returns:
        FLOPs per token
    """
    flops_per_layer = layer_flops(d_model, seq_len)
    return gate * flops_per_layer


def model_flops_per_token(d_model: int, seq_len: int, n_layers: int) -> int:
    """
    Compute total FLOPs for full model per token.
    
    Args:
        d_model: Model dimension
        seq_len: Sequence length
        n_layers: Number of layers
        
    Returns:
        Total FLOPs per token
    """
    return layer_flops(d_model, seq_len) * n_layers


def compute_effective_flops(
    gate_decisions: torch.Tensor,
    d_model: int,
    seq_len: int,
    early_layers: int = 3,
    late_layers: int = 3,
    n_layers: int = 12,
) -> Dict[str, torch.Tensor]:
    """
    Compute effective FLOPs given gate decisions for middle layers.
    
    Bowl-shaped routing means early and late layers always execute.
    
    Args:
        gate_decisions: Gate values for middle layers [batch, seq_len, n_middle_layers]
        d_model: Model dimension
        seq_len: Sequence length
        early_layers: Number of always-on early layers
        late_layers: Number of always-on late layers
        n_layers: Total number of layers
        
    Returns:
        Dictionary with FLOPs statistics
    """
    flops_per_layer = layer_flops(d_model, seq_len)
    
    # Always-on FLOPs
    always_on_flops = (early_layers + late_layers) * flops_per_layer
    
    # Gated FLOPs (sum over middle layers)
    if gate_decisions.dim() == 3:
        gated_flops = gate_decisions.sum(dim=-1) * flops_per_layer
    else:
        gated_flops = gate_decisions * flops_per_layer
    
    # Total FLOPs per token
    total_flops = always_on_flops + gated_flops
    
    # Dense baseline FLOPs
    dense_flops = n_layers * flops_per_layer
    
    # Compute ratio
    flops_ratio = total_flops / dense_flops
    
    return {
        'total_flops': total_flops,
        'always_on_flops': always_on_flops,
        'gated_flops': gated_flops,
        'dense_flops': dense_flops,
        'flops_ratio': flops_ratio,
        'mean_flops_ratio': flops_ratio.mean(),
    }


def cumulative_skipped_flops(
    gate_history: torch.Tensor,
    d_model: int,
    seq_len: int,
) -> torch.Tensor:
    """
    Compute cumulative skipped FLOPs for FLOP-aware step encoding.
    
    Args:
        gate_history: Gate values up to current layer [batch, seq_len, layers_so_far]
        d_model: Model dimension
        seq_len: Sequence length
        
    Returns:
        Cumulative skipped FLOPs per token [batch, seq_len]
    """
    flops_per_layer = layer_flops(d_model, seq_len)
    
    # Count skipped layers (gate=0 means skip)
    skipped_layers = (1 - gate_history).sum(dim=-1)
    
    return skipped_layers * flops_per_layer


def flops_to_bucket(
    cumulative_flops: torch.Tensor,
    max_flops: int,
    num_buckets: int = 64,
) -> torch.Tensor:
    """
    Convert cumulative skipped FLOPs to bucket index for step embedding lookup.
    
    Args:
        cumulative_flops: Cumulative skipped FLOPs [batch, seq_len]
        max_flops: Maximum possible skipped FLOPs
        num_buckets: Number of embedding buckets
        
    Returns:
        Bucket indices [batch, seq_len]
    """
    # Normalize to [0, 1]
    normalized = cumulative_flops.float() / max(max_flops, 1)
    
    # Convert to bucket index
    bucket_idx = (normalized * (num_buckets - 1)).long()
    
    # Clamp to valid range
    bucket_idx = bucket_idx.clamp(0, num_buckets - 1)
    
    return bucket_idx


class FLOPsTracker:
    """
    Track FLOPs across training for logging and analysis.
    """
    
    def __init__(self, d_model: int, seq_len: int, n_layers: int):
        self.d_model = d_model
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.flops_per_layer = layer_flops(d_model, seq_len)
        self.dense_flops = self.flops_per_layer * n_layers
        
        # Tracking
        self.total_flops = 0
        self.total_tokens = 0
        self.flops_history = []
        
    def update(self, gate_decisions: torch.Tensor, early_layers: int, late_layers: int):
        """Update tracker with new batch of gate decisions."""
        batch_size, seq_len = gate_decisions.shape[:2]
        
        # Compute FLOPs for this batch
        always_on = (early_layers + late_layers) * self.flops_per_layer
        
        if gate_decisions.dim() == 3:
            gated = gate_decisions.sum(dim=-1).mean() * self.flops_per_layer
        else:
            gated = gate_decisions.mean() * self.flops_per_layer
            
        batch_flops = (always_on + gated) * batch_size * seq_len
        
        self.total_flops += batch_flops.item() if isinstance(batch_flops, torch.Tensor) else batch_flops
        self.total_tokens += batch_size * seq_len
        
        ratio = (always_on + gated) / self.dense_flops
        self.flops_history.append(ratio.item() if isinstance(ratio, torch.Tensor) else ratio)
        
    def get_stats(self) -> Dict[str, float]:
        """Get current statistics."""
        return {
            'total_flops': self.total_flops,
            'total_tokens': self.total_tokens,
            'avg_flops_per_token': self.total_flops / max(self.total_tokens, 1),
            'avg_flops_ratio': sum(self.flops_history) / max(len(self.flops_history), 1),
            'dense_flops_per_token': self.dense_flops,
        }
    
    def reset(self):
        """Reset tracker."""
        self.total_flops = 0
        self.total_tokens = 0
        self.flops_history = []


# Convenience function for paper claims
def format_flops(flops: int) -> str:
    """Format FLOPs for human readability."""
    if flops >= 1e12:
        return f"{flops/1e12:.2f}T"
    elif flops >= 1e9:
        return f"{flops/1e9:.2f}G"
    elif flops >= 1e6:
        return f"{flops/1e6:.2f}M"
    elif flops >= 1e3:
        return f"{flops/1e3:.2f}K"
    else:
        return str(flops)
