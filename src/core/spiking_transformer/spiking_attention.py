# 📁 src/core/spiking_transformer/spiking_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import math

class SpikingSelfAttention(nn.Module):
    """Spiking Self-Attention با نورون‌های LIF"""
    
    def __init__(self, d_model: int = 64, n_heads: int = 4, seq_len: int = 10, 
                 spike_threshold: float = 1.0, decay: float = 0.9):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.head_dim = d_model // n_heads
        
        # پارامترهای اسپایکینگ
        self.spike_threshold = spike_threshold
        self.decay = decay
        
        # لایه‌های خطی برای Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        # membrane potentials
        self.register_buffer('membrane_q', torch.zeros(1, seq_len, d_model))
        self.register_buffer('membrane_k', torch.zeros(1, seq_len, d_model))
        self.register_buffer('membrane_v', torch.zeros(1, seq_len, d_model))
        
        # برای مانیتورینگ اسپایک‌ها
        self.spike_counts = {
            'q_spikes': 0, 'k_spikes': 0, 'v_spikes': 0
        }
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # محاسبه Q, K, V
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        
        # اعمال اسپایکینگ
        Q_spikes = self._lif_neurons(Q, 'q')
        K_spikes = self._lif_neurons(K, 'k') 
        V_spikes = self._lif_neurons(V, 'v')
        
        # تغییر شکل برای multi-head attention
        Q_spikes = Q_spikes.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K_spikes = K_spikes.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V_spikes = V_spikes.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # محاسبه attention scores
        attn_scores = torch.matmul(Q_spikes, K_spikes.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # اعمال attention به values
        attn_output = torch.matmul(attn_weights, V_spikes)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # لایه خروجی
        output = self.w_o(attn_output)
        
        return output
    
    def _lif_neurons(self, x: torch.Tensor, neuron_type: str) -> torch.Tensor:
        """Leaky Integrate-and-Fire neurons"""
        # انتخاب membrane مناسب
        if neuron_type == 'q':
            membrane = self.membrane_q
        elif neuron_type == 'k':
            membrane = self.membrane_k
        else:  # 'v'
            membrane = self.membrane_v
        
        # update membrane potential
        membrane = self.decay * membrane + x
        
        # generate spikes
        spikes = (membrane >= self.spike_threshold).float()
        
        # reset membrane
        membrane = membrane * (1 - spikes)
        
        # update buffer
        if neuron_type == 'q':
            self.membrane_q = membrane
            self.spike_counts['q_spikes'] += spikes.sum().item()
        elif neuron_type == 'k':
            self.membrane_k = membrane
            self.spike_counts['k_spikes'] += spikes.sum().item()
        else:
            self.membrane_v = membrane
            self.spike_counts['v_spikes'] += spikes.sum().item()
        
        return spikes
    
    def reset_states(self):
        """ریست حالت‌های نورون‌ها"""
        self.membrane_q.zero_()
        self.membrane_k.zero_()
        self.membrane_v.zero_()
        self.spike_counts = {'q_spikes': 0, 'k_spikes': 0, 'v_spikes': 0}
    
    def get_spike_statistics(self) -> dict:
        """آمار اسپایک‌ها"""
        total_spikes = sum(self.spike_counts.values())
        return {
            'total_spikes': total_spikes,
            'spike_rates': {k: v / (self.seq_len * self.d_model) 
                           for k, v in self.spike_counts.items()},
            **self.spike_counts
        }
