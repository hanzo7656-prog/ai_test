# 📁 src/core/spiking_transformer/spike_manager.py

import torch
import torch.nn as nn
from typing import List, Dict
from .transformer_block import SpikingTransformerBlock

class SpikeManager:
    """مدیریت اسپایک‌ها در سراسر شبکه"""
    
    def __init__(self):
        self.spike_history = []
        self.memory_usage = []
        
    def record_spikes(self, block_stats: Dict, timestamp: int):
        """ثبت آمار اسپایک‌ها"""
        record = {
            'timestamp': timestamp,
            'total_spikes': block_stats.get('total_spikes', 0),
            'spike_rates': block_stats.get('spike_rates', {}),
            'memory_used': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        }
        self.spike_history.append(record)
        
        # حفظ فقط 1000 رکورد اخیر
        if len(self.spike_history) > 1000:
            self.spike_history.pop(0)
    
    def get_spike_summary(self, window: int = 100) -> Dict:
        """خلاصه آمار اسپایک‌ها"""
        if not self.spike_history:
            return {}
            
        recent_history = self.spike_history[-window:]
        
        total_spikes = sum(record['total_spikes'] for record in recent_history)
        avg_spike_rate = total_spikes / len(recent_history) if recent_history else 0
        
        return {
            'window_size': len(recent_history),
            'total_spikes': total_spikes,
            'average_spikes_per_step': avg_spike_rate,
            'recent_spike_rates': recent_history[-1]['spike_rates'] if recent_history else {},
            'memory_efficiency': self._calculate_memory_efficiency()
        }
    
    def _calculate_memory_efficiency(self) -> float:
        """محاسبه کارایی حافظه بر اساس اسپایک‌ها"""
        if not self.spike_history:
            return 0.0
            
        recent = self.spike_history[-10:]  # 10 step اخیر
        if not recent:
            return 0.0
            
        total_computation = len(recent) * 1000  # تخمین محاسبات کامل
        actual_computation = sum(record['total_spikes'] for record in recent)
        
        return actual_computation / total_computation if total_computation > 0 else 0.0
