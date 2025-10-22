# 📁 tests/unit/core/test_sparse_network.py

import pytest
import torch
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.core.spiking_transformer.spiking_attention import SpikingSelfAttention
from src.core.spiking_transformer.spiking_ffn import SpikingFeedForward
from src.core.spiking_transformer.transformer_block import SpikingTransformerBlock
from src.core.optimization.sparse_optimizer import SparseOptimizer

class TestSparseNetwork:
    """تست شبکه اسپارس و Spiking Transformer"""
    
    def setup_method(self):
        self.d_model = 64
        self.n_heads = 4
        self.seq_len = 10
        self.batch_size = 2
        
        self.attention = SpikingSelfAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            seq_len=self.seq_len
        )
        
        self.ffn = SpikingFeedForward(d_model=self.d_model, d_ff=256)
        self.transformer_block = SpikingTransformerBlock(
            d_model=self.d_model,
            n_heads=self.n_heads,
            seq_len=self.seq_len
        )
        
        self.sparse_optimizer = SparseOptimizer(sparsity_level=0.9)
    
    def test_spiking_attention_forward(self):
        """تست forward pass attention اسپایکینگ"""
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        output = self.attention(x)
        
        assert output.shape == (self.batch_size, self.seq_len, self.d_model)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_spiking_ffn_forward(self):
        """تست forward pass FFN اسپایکینگ"""
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        output = self.ffn(x)
        
        assert output.shape == (self.batch_size, self.seq_len, self.d_model)
        assert not torch.isnan(output).any()
    
    def test_transformer_block_forward(self):
        """تست forward pass بلوک Transformer کامل"""
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        output = self.transformer_block(x)
        
        assert output.shape == (self.batch_size, self.seq_len, self.d_model)
        assert not torch.isnan(output).any()
    
    def test_spike_generation(self):
        """تست تولید اسپایک‌ها"""
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # اجرای forward برای تولید اسپایک
        self.attention(x)
        spike_stats = self.attention.get_spike_statistics()
        
        assert 'total_spikes' in spike_stats
        assert 'spike_rates' in spike_stats
        assert spike_stats['total_spikes'] >= 0
    
    def test_membrane_dynamics(self):
        """تست دینامیک membrane"""
        # تست reset states
        self.attention.reset_states()
        
        # membrane باید صفر شده باشد
        assert torch.all(self.attention.membrane_q == 0)
        assert torch.all(self.attention.membrane_k == 0)
        assert torch.all(self.attention.membrane_v == 0)
    
    def test_sparse_optimization(self):
        """تست بهینه‌سازی اسپارس"""
        # ایجاد یک مدل ساده برای تست
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)
        
        model = SimpleModel()
        
        # محاسبه اسپارسیتی قبل از بهینه‌سازی
        sparsity_before = self.sparse_optimizer.calculate_sparsity(model)
        
        # اعمال اسپارسیتی
        self.sparse_optimizer.apply_sparsity(model)
        
        # محاسبه اسپارسیتی بعد از بهینه‌سازی
        sparsity_after = self.sparse_optimizer.calculate_sparsity(model)
        
        assert sparsity_after['sparsity_percentage'] >= sparsity_before['sparsity_percentage']
        assert sparsity_after['zero_parameters'] >= sparsity_before['zero_parameters']
    
    def test_memory_efficiency(self):
        """تست کارایی حافظه"""
        import psutil
        process = psutil.Process()
        
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # اجرای چندین inference
        x = torch.randn(4, 10, 64)
        for _ in range(100):
            _ = self.transformer_block(x)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # افزایش حافظه باید معقول باشد
        memory_increase = memory_after - memory_before
        assert memory_increase < 100  # کمتر از 100MB
    
    def test_gradient_flow(self):
        """تست جریان گرادیان"""
        x = torch.randn(2, 10, 64, requires_grad=True)
        output = self.transformer_block(x)
        
        # ایجاد loss مصنوعی
        loss = output.sum()
        loss.backward()
        
        # گرادیان‌ها باید وجود داشته باشند
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
