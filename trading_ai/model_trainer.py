import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Optional
import logging
import joblib
from datetime import datetime

from database_manager import trading_db
from real_time_analyzer import market_analyzer
from advanced_technical_engine import technical_engine
from sparse_technical_analyzer import SparseTechnicalNetwork, SparseConfig

logger = logging.getLogger(__name__)

class SparseModelTrainer:
    """آموزش‌دهنده مدل اسپارس تحلیل تکنیکال"""
    
    def __init__(self):
        self.config = SparseConfig()
        self.model = None
        self.scaler = None
        
    def train_technical_analysis(self, symbols: List[str], epochs: int = 30) -> Dict:
        """آموزش مدل اسپارس روی نمادهای مختلف"""
        try:
            all_sequences = []
            all_labels = []
            
            # جمع‌آوری داده‌های آموزشی از تمام نمادها
            for symbol in symbols:
                sequences, labels = technical_engine.prepare_training_data(symbol)
                if sequences is not None:
                    all_sequences.append(sequences)
                    all_labels.append(labels)
                    logger.info(f"📥 داده‌های {symbol} اضافه شد: {len(sequences)} نمونه")
            
            if not all_sequences:
                logger.error("❌ هیچ داده آموزشی یافت نشد")
                return {}
            
            # ترکیب داده‌های تمام نمادها
            X_train = np.concatenate(all_sequences, axis=0)
            y_train = np.concatenate(all_labels, axis=0)
            
            logger.info(f"📊 داده‌های آموزشی ترکیب شد: {X_train.shape}")
            
            # ایجاد مدل اسپارس
            self.model = SparseTechnicalNetwork(self.config)
            
            # آموزش مدل
            training_results = self._train_model(X_train, y_train, epochs)
            
            # ذخیره مدل آموزش دیده
            self._save_model(symbols[0], training_results)
            
            logger.info("✅ آموزش مدل اسپارس تکمیل شد")
            return training_results
            
        except Exception as e:
            logger.error(f"❌ خطا در آموزش مدل: {e}")
            return {}
    
    def _train_model(self, X: np.ndarray, y: np.ndarray, epochs: int) -> Dict:
        """آموزش مدل با داده‌های آماده شده"""
        
        # تقسیم داده به آموزش و validation
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # تبدیل به تانسور
        X_train_tensor = torch.FloatTensor(X_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_train_tensor = torch.LongTensor(y_train)
        y_val_tensor = torch.LongTensor(y_val)
        
        # ایجاد DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # تنظیمات آموزش
        optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # آموزش
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            # فاز آموزش
            self.model.train()
            epoch_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                loss = criterion(outputs['trend_strength'], batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # فاز Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs['trend_strength'], y_val_tensor)
            
            train_losses.append(epoch_loss / len(train_loader))
            val_losses.append(val_loss.item())
            
            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"⏹️ توقف زودهنگام در دوره {epoch}")
                break
            
            if epoch % 5 == 0:
                logger.info(f"📊 دوره {epoch}: Train_Loss={train_losses[-1]:.4f}, Val_Loss={val_losses[-1]:.4f}")
        
        # بارگذاری بهترین مدل
        self.model.load_state_dict(best_model_state)
        
        # محاسبه دقت نهایی
        with torch.no_grad():
            val_outputs = self.model(X_val_tensor)
            _, predicted = torch.max(val_outputs['trend_strength'], 1)
            accuracy = (predicted == y_val_tensor).float().mean().item()
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_accuracy': accuracy,
            'best_val_loss': best_val_loss,
            'epochs_trained': epoch + 1
        }
    
    def _save_model(self, symbol: str, results: Dict):
        """ذخیره مدل آموزش دیده"""
        import os
        os.makedirs("./models", exist_ok=True)
        
        model_path = f"./models/sparse_technical_{symbol}.pth"
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'training_results': results,
            'config': self.config.__dict__,
            'trained_at': datetime.now().isoformat(),
            'model_type': 'sparse_technical_analyzer'
        }, model_path)
        
        logger.info(f"💾 مدل اسپارس ذخیره شد: {model_path}")
    
    def load_model(self, model_path: str):
        """بارگذاری مدل آموزش دیده"""
        checkpoint = torch.load(model_path)
        
        self.config = SparseConfig(**checkpoint['config'])
        self.model = SparseTechnicalNetwork(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"📥 مدل اسپارس بارگذاری شد: {model_path}")

# ایجاد نمونه گلوبال
model_trainer = SparseModelTrainer()
