# 📁 src/core/spiking_transformer/spiking_ffn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpikingFeedForward(nn.Module):
    """لایه Feed-Forward اسپایکینگ"""
    
    def __init__(self, d_model: int = 64, d_ff: int = 256, 
                 spike_threshold: float = 0.5, decay: float = 0.8):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # لایه‌های خطی
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # پارامترهای اسپایکینگ
        self.spike_threshold = spike_threshold
        self.decay = decay
        
        # membrane potentials
        self.register_buffer('membrane', torch.zeros(1, 1, d_ff))
        
        self.spike_count = 0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # لایه اول با اسپایکینگ
        hidden = self.linear1(x)
        hidden_spikes = self._lif_activation(hidden)
        
        # لایه دوم (بدون اسپایکینگ)
        output = self.linear2(hidden_spikes)
        
        return output
    
    def _lif_activation(self, x: torch.Tensor) -> torch.Tensor:
        """فعال‌سازی LIF برای لایه پنهان"""
        # update membrane potential
        self.membrane = self.decay * self.membrane + x
        
        # generate spikes
        spikes = (self.membrane >= self.spike_threshold).float()
        
        # reset membrane
        self.membrane = self.membrane * (1 - spikes)
        
        self.spike_count += spikes.sum().item()
        
        return spikes
    
    def reset_states(self):
        self.membrane.zero_()
        self.spike_count = 0
    
    def get_spike_stats(self) -> dict:
        return {
            'ffn_spike_count': self.spike_count,
            'ffn_spike_rate': self.spike_count / (self.d_ff * self.membrane.shape[1])
        }
