"""
Elastic-Depth Pretraining (EDP) Transformer.

The main model that implements:
- Second-order exit signal
- Threshold-based elastic routing
- Bowl-shaped layer allocation
- FLOP-aware step encoding
- Warp-forward mechanism
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple

from .components import (
    TokenEmbedding,
    PositionalEncoding,
    TransformerBlock,
)
from ..signals import SignalComputer
from ..routing import ElasticRouter
from ..utils.flops import layer_flops, compute_effective_flops


class EDPTransformerBlock(nn.Module):
    """
    Transformer block with elastic depth support.
    
    Extends base block with:
    - Hidden state output for signal computation
    - Gated execution
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_expansion: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 256,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Pre-LN
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # FFN
        d_ffn = d_model * ffn_expansion
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model, bias=False),
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer('causal_mask', mask)
        
    def forward(
        self,
        x: torch.Tensor,
        gate: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional gating.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            gate: Binary gate [batch, seq_len] or None (no gating)
            attention_mask: Padding mask
            
        Returns:
            output: Block output [batch, seq_len, d_model]
        """
        seq_len = x.shape[1]
        
        # Get causal mask
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        
        # Attention with Pre-LN
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed,
            attn_mask=causal_mask,
            need_weights=False,
        )
        attn_out = self.dropout(attn_out)
        
        # FFN with Pre-LN
        x_attn = x + attn_out
        normed = self.norm2(x_attn)
        ffn_out = self.ffn(normed)
        ffn_out = self.dropout(ffn_out)
        
        out = x_attn + ffn_out
        
        # Apply gating if provided
        if gate is not None:
            # gate: [batch, seq_len] -> [batch, seq_len, 1]
            gate_expanded = gate.unsqueeze(-1)
            # Gated output: skip if gate=0, compute if gate=1
            out = gate_expanded * out + (1 - gate_expanded) * x
            
        return out


class EDPTransformer(nn.Module):
    """
    Elastic-Depth Pretraining Transformer.
    
    Key features:
    - Second-order exit signal for routing decisions
    - Learned per-layer thresholds
    - Bowl-shaped routing (early/late always on, middle gated)
    - FLOP-aware step encoding for warp-forward
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 12,
        ffn_expansion: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        # EDP-specific
        early_layers: int = 3,
        late_layers: int = 3,
        use_second_order_signal: bool = True,
        learned_threshold: bool = True,
        initial_threshold: float = 0.1,
        use_flop_aware_encoding: bool = True,
        use_warp_forward: bool = True,
        num_step_buckets: int = 64,
        max_skip_layers: int = 4,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.early_layers = early_layers
        self.late_layers = late_layers
        self.use_warp_forward = use_warp_forward
        
        # Embeddings
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            EDPTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                ffn_expansion=ffn_expansion,
                dropout=dropout,
                max_seq_len=max_seq_len,
            )
            for _ in range(n_layers)
        ])
        
        # Signal computer
        self.signal_computer = SignalComputer(
            use_second_order=use_second_order_signal,
            normalize=True,
        )
        
        # Elastic router
        self.router = ElasticRouter(
            d_model=d_model,
            n_layers=n_layers,
            early_layers=early_layers,
            late_layers=late_layers,
            learned_threshold=learned_threshold,
            initial_threshold=initial_threshold,
            use_flop_aware_encoding=use_flop_aware_encoding,
            num_step_buckets=num_step_buckets,
            max_skip_layers=max_skip_layers,
            seq_len=max_seq_len,
        )
        
        # Output
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.token_embedding.embedding.weight
        
        # FLOPs tracking
        self.flops_per_layer = layer_flops(d_model, max_seq_len)
        
        # Initialize
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
            
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_routing_info: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with elastic depth routing.
        
        Args:
            input_ids: Token indices [batch, seq_len]
            attention_mask: Padding mask [batch, seq_len]
            labels: Target tokens for loss [batch, seq_len]
            return_routing_info: If True, return detailed routing statistics
            
        Returns:
            Dictionary with logits, loss, routing info
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embed
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)
        
        # Reset signal computer for new sequence
        self.signal_computer.reset()
        
        # Track routing decisions
        all_gates = []
        all_soft_gates = []
        all_signals = []
        
        # Cumulative skipped FLOPs per token
        cumulative_skipped_flops = torch.zeros(batch_size, seq_len, device=device)
        
        # Previous hidden state for signal computation
        h_prev = x.clone()
        
        # Process each layer
        for layer_idx, layer in enumerate(self.layers):
            # Compute signal (second-order or first-order)
            signal = self.signal_computer(x, h_prev)
            all_signals.append(signal)
            
            # Get routing decision
            gate, soft_gate, cumulative_skipped_flops = self.router(
                signal=signal,
                layer_idx=layer_idx,
                cumulative_skipped_flops=cumulative_skipped_flops,
            )
            
            all_gates.append(gate)
            all_soft_gates.append(soft_gate)
            
            # Store current hidden state
            h_prev = x.clone()
            
            # Apply warp-forward step encoding if using it
            if self.use_warp_forward and not self.router.is_always_on(layer_idx):
                x = self.router.apply_step_encoding(x, layer_idx, cumulative_skipped_flops)
            
            # Execute layer with gating
            if self.router.is_always_on(layer_idx):
                # Always-on layer: no gating
                x = layer(x, gate=None, attention_mask=attention_mask)
            else:
                # Gated layer
                x = layer(x, gate=gate, attention_mask=attention_mask)
        
        # Output
        x = self.norm(x)
        logits = self.lm_head(x)
        
        # Stack gates for loss computation
        gates_stacked = torch.stack(all_gates, dim=-1)  # [batch, seq, n_layers]
        soft_gates_stacked = torch.stack(all_soft_gates, dim=-1)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        # Build output
        output = {
            'logits': logits,
            'loss': loss,
            'gates': gates_stacked,
            'soft_gates': soft_gates_stacked,
            'cumulative_skipped_flops': cumulative_skipped_flops,
        }
        
        if return_routing_info:
            # Compute detailed routing statistics
            n_middle_layers = self.n_layers - self.early_layers - self.late_layers
            
            # Extract middle layer gates
            middle_gates = gates_stacked[..., self.early_layers:self.n_layers - self.late_layers]
            
            # Depth used per token
            depth_used = gates_stacked.sum(dim=-1)
            
            # FLOPs ratio
            flops_info = compute_effective_flops(
                middle_gates,
                self.d_model,
                seq_len,
                self.early_layers,
                self.late_layers,
                self.n_layers,
            )
            
            output['routing_info'] = {
                'depth_used': depth_used,
                'middle_gates': middle_gates,
                'gate_ratio_per_layer': gates_stacked.float().mean(dim=(0, 1)),
                'signals': torch.stack(all_signals, dim=-1),
                'thresholds': self.router.router.get_thresholds(),
                **flops_info,
            }
            
        return output
    
    def compute_token_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute per-token entropy from logits.
        
        Used for entropy-aware sparsity loss.
        
        Args:
            logits: Model logits [batch, seq_len, vocab_size]
            
        Returns:
            entropy: Per-token entropy [batch, seq_len]
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        
        # Normalize by max entropy (log vocab_size)
        max_entropy = torch.log(torch.tensor(self.vocab_size, dtype=torch.float, device=logits.device))
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Count parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.pos_encoding.pos_embedding.weight.numel()
        return n_params
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Generate tokens autoregressively.
        
        Args:
            input_ids: Starting tokens [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            
        Returns:
            Generated tokens [batch, seq_len + max_new_tokens]
            Generation statistics
        """
        self.eval()
        
        all_flops_ratios = []
        
        for _ in range(max_new_tokens):
            # Truncate to max_seq_len
            input_truncated = input_ids[:, -self.max_seq_len:]
            
            # Forward pass
            with torch.no_grad():
                outputs = self.forward(input_truncated, return_routing_info=True)
                
            # Track FLOPs
            all_flops_ratios.append(outputs['routing_info']['mean_flops_ratio'].item())
            
            # Get next token logits
            next_logits = outputs['logits'][:, -1, :] / temperature
            
            # Apply filtering
            if top_k is not None:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')
                
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float('-inf')
                
            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
        stats = {
            'avg_flops_ratio': sum(all_flops_ratios) / len(all_flops_ratios),
            'flops_ratios': all_flops_ratios,
        }
        
        return input_ids, stats


def create_edp_model(config) -> EDPTransformer:
    """Create EDP model from config."""
    return EDPTransformer(
        vocab_size=config.model.vocab_size,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        ffn_expansion=config.model.ffn_expansion,
        dropout=config.model.dropout,
        max_seq_len=config.model.max_seq_len,
        # EDP-specific
        early_layers=config.edp.early_layers,
        late_layers=config.edp.late_layers,
        use_second_order_signal=config.edp.use_second_order_signal,
        learned_threshold=config.edp.learned_threshold,
        initial_threshold=config.edp.initial_threshold,
        use_flop_aware_encoding=config.edp.use_flop_aware_encoding,
        use_warp_forward=config.edp.use_warp_forward,
        num_step_buckets=config.edp.num_step_embeddings,
        max_skip_layers=config.edp.max_skip_layers,
    )
