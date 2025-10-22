# 📁 src/core/spiking_transformer/transformer_block.py

import torch.nn as nn
from .spiking_attention import SpikingSelfAttention
from .spiking_ffn import SpikingFeedForward

class SpikingTransformerBlock(nn.Module):
    """بلوک کامل Spiking Transformer"""
    
    def __init__(self, d_model: int = 64, n_heads: int = 4, seq_len: int = 10,
                 d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        
        self.attention = SpikingSelfAttention(d_model, n_heads, seq_len)
        self.ffn = SpikingFeedForward(d_model, d_ff)
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention با residual connection
        attn_output = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward با residual connection
        ff_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
    
    def reset_states(self):
        self.attention.reset_states()
        self.ffn.reset_states()
    
    def get_spike_statistics(self) -> dict:
        attn_stats = self.attention.get_spike_statistics()
        ffn_stats = self.ffn.get_spike_stats()
        return {**attn_stats, **ffn_stats}
