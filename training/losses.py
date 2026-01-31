"""
Loss Functions for Elastic-Depth Pretraining.

Implements the full objective:
L = L_LM + λ1 * L_sparsity + λ2 * L_budget

Where:
- L_LM: Standard language modeling cross-entropy
- L_sparsity: Encourages binary (easy vs hard) depth usage
- L_budget: Enforces global compute budget discipline
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class SparsityLoss(nn.Module):
    """
    Sparsity loss that encourages sharp separation between easy and hard tokens.
    
    L_sparsity = (1/T) * Σ_t (d_t / D)²
    
    Where d_t is depth used by token t, D is total depth.
    
    Squared penalty:
    - Strongly penalizes tokens that "refuse to skip"
    - Encourages binary behavior: either skip or compute fully
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        gates: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute sparsity loss.
        
        Args:
            gates: Gate decisions [batch, seq_len, n_layers] or [batch, seq_len]
            attention_mask: Mask for valid tokens [batch, seq_len]
            
        Returns:
            Scalar sparsity loss
        """
        # Sum gates across layers to get depth used per token
        if gates.dim() == 3:
            depth_used = gates.sum(dim=-1)  # [batch, seq_len]
            total_depth = gates.shape[-1]
        else:
            depth_used = gates
            total_depth = 1.0
            
        # Normalize by total depth
        depth_ratio = depth_used / total_depth
        
        # Squared penalty
        sparsity = depth_ratio ** 2
        
        # Apply mask if provided
        if attention_mask is not None:
            sparsity = sparsity * attention_mask
            n_tokens = attention_mask.sum()
        else:
            n_tokens = sparsity.numel()
            
        # Mean over valid tokens
        loss = sparsity.sum() / n_tokens.clamp(min=1)
        
        return loss


class EntropyAwareSparsityLoss(nn.Module):
    """
    Entropy-aware sparsity loss (optional, FAIR-coded).
    
    L = (1/T) * Σ_t (1 - Ĥ_t) * (d_t / D)
    
    Where Ĥ_t is normalized token entropy.
    
    Interpretation:
    - High-confidence tokens (low entropy) → must skip
    - Low-confidence tokens (high entropy) → allowed to compute
    """
    
    def __init__(self, vocab_size: int):
        super().__init__()
        self.max_entropy = torch.log(torch.tensor(vocab_size, dtype=torch.float))
        
    def forward(
        self,
        gates: torch.Tensor,
        logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute entropy-aware sparsity loss.
        
        Args:
            gates: Gate decisions [batch, seq_len, n_layers]
            logits: Model logits [batch, seq_len, vocab_size]
            attention_mask: Mask for valid tokens [batch, seq_len]
            
        Returns:
            Scalar loss
        """
        # Compute per-token entropy
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # [batch, seq_len]
        
        # Normalize entropy
        normalized_entropy = entropy / self.max_entropy.to(entropy.device)
        normalized_entropy = normalized_entropy.clamp(0, 1)
        
        # Depth ratio
        if gates.dim() == 3:
            depth_used = gates.sum(dim=-1)
            total_depth = gates.shape[-1]
        else:
            depth_used = gates
            total_depth = 1.0
            
        depth_ratio = depth_used / total_depth
        
        # Entropy-weighted sparsity
        # (1 - entropy) means high weight for low entropy (confident) tokens
        sparsity = (1 - normalized_entropy) * depth_ratio
        
        # Apply mask if provided
        if attention_mask is not None:
            sparsity = sparsity * attention_mask
            n_tokens = attention_mask.sum()
        else:
            n_tokens = sparsity.numel()
            
        loss = sparsity.sum() / n_tokens.clamp(min=1)
        
        return loss


class BudgetLoss(nn.Module):
    """
    Budget loss that enforces global compute discipline.
    
    L_budget = |avg_gate - target_ratio|
    
    Typical targets:
    - Training: 0.6
    - Evaluation: 0.5
    """
    
    def __init__(self, target_ratio: float = 0.6):
        super().__init__()
        self.target_ratio = target_ratio
        
    def forward(
        self,
        gates: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute budget loss.
        
        Args:
            gates: Gate decisions [batch, seq_len, n_layers] or [batch, seq_len]
            attention_mask: Mask for valid tokens [batch, seq_len]
            
        Returns:
            Scalar budget loss
        """
        # Compute average gate activation
        if attention_mask is not None:
            if gates.dim() == 3:
                # Expand mask for layers
                mask = attention_mask.unsqueeze(-1)
                avg_gate = (gates * mask).sum() / (mask.sum() * gates.shape[-1])
            else:
                avg_gate = (gates * attention_mask).sum() / attention_mask.sum()
        else:
            avg_gate = gates.mean()
            
        # L1 distance from target
        loss = torch.abs(avg_gate - self.target_ratio)
        
        return loss
    
    def set_target(self, target_ratio: float):
        """Update target ratio."""
        self.target_ratio = target_ratio


class EDPLoss(nn.Module):
    """
    Complete EDP loss combining LM, sparsity, and budget losses.
    
    L = L_LM + λ1 * L_sparsity + λ2 * L_budget
    
    With warmup schedule for sparsity loss.
    """
    
    def __init__(
        self,
        lambda_sparsity: float = 0.1,
        lambda_budget: float = 0.05,
        target_compute_ratio: float = 0.6,
        sparsity_warmup_fraction: float = 0.2,
        use_entropy_aware: bool = False,
        vocab_size: int = 50257,
    ):
        super().__init__()
        
        self.lambda_sparsity = lambda_sparsity
        self.lambda_budget = lambda_budget
        self.sparsity_warmup_fraction = sparsity_warmup_fraction
        
        # Loss components
        if use_entropy_aware:
            self.sparsity_loss = EntropyAwareSparsityLoss(vocab_size)
        else:
            self.sparsity_loss = SparsityLoss()
            
        self.budget_loss = BudgetLoss(target_compute_ratio)
        self.use_entropy_aware = use_entropy_aware
        
        # Training state
        self.current_step = 0
        self.total_steps = 1
        
    def set_training_steps(self, total_steps: int):
        """Set total training steps for warmup schedule."""
        self.total_steps = total_steps
        
    def step(self):
        """Advance step counter."""
        self.current_step += 1
        
    def get_sparsity_weight(self) -> float:
        """Get current sparsity loss weight with warmup."""
        warmup_steps = int(self.sparsity_warmup_fraction * self.total_steps)
        
        if self.current_step < warmup_steps:
            # Linear warmup
            return self.lambda_sparsity * (self.current_step / warmup_steps)
        else:
            return self.lambda_sparsity
        
    def forward(
        self,
        lm_loss: torch.Tensor,
        gates: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute full EDP loss.
        
        Args:
            lm_loss: Language modeling loss (scalar)
            gates: Gate decisions [batch, seq_len, n_layers]
            logits: Model logits (required if entropy-aware)
            attention_mask: Padding mask [batch, seq_len]
            
        Returns:
            Dictionary with all loss components and total
        """
        # Sparsity loss
        if self.use_entropy_aware:
            assert logits is not None, "Logits required for entropy-aware loss"
            sparsity = self.sparsity_loss(gates, logits, attention_mask)
        else:
            sparsity = self.sparsity_loss(gates, attention_mask)
            
        # Budget loss
        budget = self.budget_loss(gates, attention_mask)
        
        # Get current sparsity weight (with warmup)
        sparsity_weight = self.get_sparsity_weight()
        
        # Total loss
        total = lm_loss + sparsity_weight * sparsity + self.lambda_budget * budget
        
        return {
            'total_loss': total,
            'lm_loss': lm_loss,
            'sparsity_loss': sparsity,
            'budget_loss': budget,
            'sparsity_weight': torch.tensor(sparsity_weight),
        }


def compute_perplexity(loss: torch.Tensor) -> torch.Tensor:
    """Compute perplexity from cross-entropy loss."""
    return torch.exp(loss)
