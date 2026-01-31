"""
Second-Order Exit Signal for Elastic-Depth Pretraining.

This module implements the second-order residual dynamics signal,
which is the key innovation over first-order DiffSkip.

For token t at layer i:
    delta_i = ||h_i - h_{i-1}||   (first-order: velocity)
    accel_i = |delta_i - delta_{i-1}|  (second-order: acceleration)

Interpretation:
- High acceleration → representation still evolving
- Low acceleration → steady state → safe to skip

This is far more stable than magnitude-only gating because:
1. Tokens that oscillate have high velocity but low acceleration
2. Tokens converging smoothly have decreasing acceleration
3. Natural stopping criterion when accel ≈ 0
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict


class FirstOrderSignal(nn.Module):
    """
    First-order (velocity) signal: ||h_i - h_{i-1}||
    
    Used as baseline comparison in ablations.
    """
    
    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
        
    def forward(
        self,
        h_current: torch.Tensor,
        h_previous: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute first-order signal.
        
        Args:
            h_current: Current hidden state [batch, seq_len, d_model]
            h_previous: Previous hidden state [batch, seq_len, d_model]
            
        Returns:
            delta: First-order signal [batch, seq_len]
        """
        # Compute L2 norm of difference
        delta = (h_current - h_previous).norm(dim=-1)
        
        if self.normalize:
            # Normalize by hidden state norm to make scale-invariant
            h_norm = h_current.norm(dim=-1).clamp(min=1e-8)
            delta = delta / h_norm
            
        return delta


class SecondOrderSignal(nn.Module):
    """
    Second-order (acceleration) signal: |delta_i - delta_{i-1}|
    
    This is the key innovation of EDP over DiffSkip.
    """
    
    def __init__(self, normalize: bool = True, eps: float = 1e-8):
        super().__init__()
        self.normalize = normalize
        self.eps = eps
        self.first_order = FirstOrderSignal(normalize=normalize)
        
    def forward(
        self,
        h_current: torch.Tensor,
        h_previous: torch.Tensor,
        delta_previous: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute second-order (acceleration) signal.
        
        Args:
            h_current: Current hidden state [batch, seq_len, d_model]
            h_previous: Previous hidden state [batch, seq_len, d_model]
            delta_previous: Previous delta (velocity) [batch, seq_len], or None for first layer
            
        Returns:
            acceleration: Second-order signal [batch, seq_len]
            delta_current: Current delta for next layer [batch, seq_len]
        """
        # Compute current velocity
        delta_current = self.first_order(h_current, h_previous)
        
        if delta_previous is None:
            # First gated layer: acceleration is just the delta
            acceleration = delta_current
        else:
            # Compute acceleration as change in velocity
            acceleration = (delta_current - delta_previous).abs()
            
        return acceleration, delta_current


class ResidualMagnitudeSignal(nn.Module):
    """
    Simple hidden state magnitude signal.
    
    Used as baseline comparison - this is what naive approaches might use.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute hidden state magnitude.
        
        Args:
            h: Hidden state [batch, seq_len, d_model]
            
        Returns:
            magnitude: L2 norm [batch, seq_len]
        """
        return h.norm(dim=-1)


class AttentionDiffSignal(nn.Module):
    """
    DiffSkip-style signal: ||Attn(x) - x||
    
    Original DiffSkip signal for comparison.
    """
    
    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
        
    def forward(
        self,
        attn_output: torch.Tensor,
        attn_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attention difference signal.
        
        Args:
            attn_output: Output of attention layer [batch, seq_len, d_model]
            attn_input: Input to attention layer [batch, seq_len, d_model]
            
        Returns:
            diff: Attention difference signal [batch, seq_len]
        """
        diff = (attn_output - attn_input).norm(dim=-1)
        
        if self.normalize:
            input_norm = attn_input.norm(dim=-1).clamp(min=1e-8)
            diff = diff / input_norm
            
        return diff


class SignalComputer(nn.Module):
    """
    Unified signal computation for EDP.
    
    Manages state across layers and computes the appropriate signal type.
    """
    
    def __init__(
        self,
        use_second_order: bool = True,
        normalize: bool = True,
    ):
        super().__init__()
        self.use_second_order = use_second_order
        
        if use_second_order:
            self.signal_fn = SecondOrderSignal(normalize=normalize)
        else:
            self.signal_fn = FirstOrderSignal(normalize=normalize)
            
        # State tracking
        self.delta_previous = None
        self.h_previous = None
        
    def reset(self):
        """Reset state for new sequence."""
        self.delta_previous = None
        self.h_previous = None
        
    def forward(
        self,
        h_current: torch.Tensor,
        h_previous: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute signal for current layer.
        
        Args:
            h_current: Current hidden state [batch, seq_len, d_model]
            h_previous: Previous hidden state (optional, uses stored if None)
            
        Returns:
            signal: Routing signal [batch, seq_len]
        """
        if h_previous is None:
            h_previous = self.h_previous
            
        if h_previous is None:
            # First layer - no previous state
            self.h_previous = h_current
            return torch.zeros(h_current.shape[0], h_current.shape[1], device=h_current.device)
            
        if self.use_second_order:
            signal, delta_current = self.signal_fn(h_current, h_previous, self.delta_previous)
            self.delta_previous = delta_current
        else:
            signal = self.signal_fn(h_current, h_previous)
            
        self.h_previous = h_current
        return signal
    
    def get_signal_stats(self, signal: torch.Tensor) -> Dict[str, float]:
        """Get statistics about the signal for logging."""
        return {
            'signal_mean': signal.mean().item(),
            'signal_std': signal.std().item(),
            'signal_min': signal.min().item(),
            'signal_max': signal.max().item(),
        }


# Factory function
def create_signal_computer(
    signal_type: str = "second_order",
    normalize: bool = True,
) -> nn.Module:
    """
    Create signal computer of specified type.
    
    Args:
        signal_type: One of "second_order", "first_order", "magnitude", "attention_diff"
        normalize: Whether to normalize signals
        
    Returns:
        Signal computation module
    """
    if signal_type == "second_order":
        return SignalComputer(use_second_order=True, normalize=normalize)
    elif signal_type == "first_order":
        return SignalComputer(use_second_order=False, normalize=normalize)
    elif signal_type == "magnitude":
        return ResidualMagnitudeSignal()
    elif signal_type == "attention_diff":
        return AttentionDiffSignal(normalize=normalize)
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")
