"""
Dense Baseline Transformer for comparison.

Frozen Spec:
- Decoder-only
- Pre-LN
- 12 layers
- d_model = 512
- n_heads = 8
- FFN expansion = 4x
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

from .components import (
    TokenEmbedding,
    PositionalEncoding,
    TransformerBlock,
)


class DenseTransformer(nn.Module):
    """
    Dense (non-elastic) transformer baseline.
    
    This is the standard transformer that EDP will be compared against.
    All tokens traverse all layers.
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
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                ffn_expansion=ffn_expansion,
                dropout=dropout,
                max_seq_len=max_seq_len,
            )
            for _ in range(n_layers)
        ])
        
        # Output
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.token_embedding.embedding.weight
        
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
        return_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token indices [batch, seq_len]
            attention_mask: Padding mask [batch, seq_len]
            labels: Target tokens for loss [batch, seq_len]
            return_hidden_states: If True, return all hidden states
            
        Returns:
            Dictionary with logits, loss (if labels provided), hidden states
        """
        # Embed
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)
        
        # Track hidden states
        hidden_states = [x] if return_hidden_states else None
        
        # Transformer layers
        for layer in self.layers:
            x, _ = layer(x, attention_mask)
            if return_hidden_states:
                hidden_states.append(x)
                
        # Output
        x = self.norm(x)
        logits = self.lm_head(x)
        
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
            
        output = {
            'logits': logits,
            'loss': loss,
        }
        
        if return_hidden_states:
            output['hidden_states'] = hidden_states
            
        return output
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
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
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Truncate to max_seq_len
            input_truncated = input_ids[:, -self.max_seq_len:]
            
            # Forward pass
            with torch.no_grad():
                outputs = self.forward(input_truncated)
                
            # Get next token logits
            next_logits = outputs['logits'][:, -1, :] / temperature
            
            # Apply filtering
            if top_k is not None:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')
                
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
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
            
        return input_ids
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Count parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.pos_encoding.pos_embedding.weight.numel()
        return n_params


def create_baseline_model(config) -> DenseTransformer:
    """Create baseline model from config."""
    return DenseTransformer(
        vocab_size=config.model.vocab_size,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        ffn_expansion=config.model.ffn_expansion,
        dropout=config.model.dropout,
        max_seq_len=config.model.max_seq_len,
    )
