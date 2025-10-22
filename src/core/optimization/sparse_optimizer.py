# 📁 src/core/optimization/sparse_optimizer.py

import torch
import torch.nn as nn
from typing import Dict

class SparseOptimizer:
    """بهینه‌ساز وزن‌های اسپارس"""
    
    def __init__(self, sparsity_level: float = 0.9):
        self.sparsity_level = sparsity_level
    
    def apply_sparsity(self, model):
        """اعمال اسپارسیتی به مدل"""
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                self._make_sparse(param)
    
    def _make_sparse(self, tensor):
        """تبدیل تنسور به اسپارس"""
        with torch.no_grad():
            # محاسبه آستانه بر اساس اسپارسیتی مورد نظر
            threshold = torch.quantile(
                torch.abs(tensor.flatten()), 
                self.sparsity_level
            )
            
            # ایجاد ماسک اسپارس
            mask = torch.abs(tensor) > threshold
            tensor.data *= mask.float()
    
    def calculate_sparsity(self, model) -> Dict:
        """محاسبه سطح اسپارسیتی مدل"""
        total_params = 0
        zero_params = 0
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                total_params += param.numel()
                zero_params += (param == 0).sum().item()
        
        sparsity = zero_params / total_params if total_params > 0 else 0
        
        return {
            'total_parameters': total_params,
            'zero_parameters': zero_params,
            'sparsity_percentage': sparsity * 100,
            'effective_parameters': total_params - zero_params
        }
