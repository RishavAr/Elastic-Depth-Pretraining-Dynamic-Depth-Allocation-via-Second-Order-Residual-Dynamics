"""
Shared model components for baseline and EDP transformers.

All components follow the frozen spec:
- Pre-LN (LayerNorm before attention/FFN)
- Multi-head self-attention with causal mask
- FFN with 4x expansion and GELU activation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with causal masking.
    
    Optimized for decoder-only transformer.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 256,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        
        # QKV projection (combined for efficiency)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask (registered as buffer for proper device handling)
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer('causal_mask', mask)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with causal self-attention.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional padding mask [batch, seq_len]
            
        Returns:
            output: Attention output [batch, seq_len, d_model]
            attn_weights: Attention weights [batch, n_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # QKV projection
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, d_head]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply padding mask if provided
        if attention_mask is not None:
            # attention_mask: [batch, seq_len], 1 = attend, 0 = mask
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        
        # Reshape and project
        out = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)
        
        return out, attn_weights


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    
    FFN(x) = GELU(xW1)W2
    
    With 4x expansion as specified.
    """
    
    def __init__(
        self,
        d_model: int,
        expansion: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_ffn = d_model * expansion
        
        self.up = nn.Linear(d_model, d_ffn, bias=False)
        self.down = nn.Linear(d_ffn, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            
        Returns:
            output: FFN output [batch, seq_len, d_model]
        """
        x = self.up(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.down(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer block with Pre-LN.
    
    Structure:
    - LayerNorm -> Attention -> Residual
    - LayerNorm -> FFN -> Residual
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
        
        # Pre-LN
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Attention
        self.attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        
        # FFN
        self.ffn = FeedForward(
            d_model=d_model,
            expansion=ffn_expansion,
            dropout=dropout,
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attn_output: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional padding mask
            return_attn_output: If True, return attention output before residual
            
        Returns:
            output: Block output [batch, seq_len, d_model]
            attn_output: Attention output if return_attn_output=True
        """
        # Attention with Pre-LN
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, attention_mask)
        attn_out = self.dropout(attn_out)
        x = x + attn_out
        
        # FFN with Pre-LN
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        ffn_out = self.dropout(ffn_out)
        x = x + ffn_out
        
        if return_attn_output:
            return x, attn_out
        return x, None


class PositionalEncoding(nn.Module):
    """
    Learned positional encoding.
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding.
        
        Args:
            x: Input embeddings [batch, seq_len, d_model]
            
        Returns:
            Output with positional encoding [batch, seq_len, d_model]
        """
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        pos_emb = self.pos_embedding(positions)
        return self.dropout(x + pos_emb)


class TokenEmbedding(nn.Module):
    """
    Token embedding with scaling.
    """
    
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        
        # Initialize
        nn.init.normal_(self.embedding.weight, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed tokens.
        
        Args:
            x: Token indices [batch, seq_len]
            
        Returns:
            Embeddings [batch, seq_len, d_model]
        """
        return self.embedding(x) * math.sqrt(self.d_model)
