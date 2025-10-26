# model_trainer.py - آموزش مدل روی داده‌های واقعی
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import logging
from typing import Dict, Tuple, List
import joblib

from database_manager import trading_db
from real_time_analyzer import market_analyzer

logger = logging.getLogger(__name__)

class RealTradingModel(nn.Module):
    """مدل تریدینگ واقعی با معماری پیشرفته"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32], dropout: float = 0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # ایجاد لایه‌های پویا
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # سرهای خروجی مختلف
        self.signal_head = nn.Linear(prev_dim, 3)  # BUY, SELL, HOLD
        self.regression_head = nn.Linear(prev_dim, 1)  # پیش‌بینی بازده
        self.confidence_head = nn.Linear(prev_dim, 1)  # اطمینان پیش‌بینی
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.feature_extractor(x)
        
        signal_logits = self.signal_head(features)
        regression_output = self.regression_head(features)
        confidence = torch.sigmoid(self.confidence_head(features))
        
        return {
            'signals': signal_logits,
            'returns': regression_output,
            'confidence': confidence.squeeze(-1)
        }

class RealModelTrainer:
    """آموزش‌دهنده مدل واقعی"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = []
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """آماده‌سازی ویژگی‌ها برای آموزش"""
        # انتخاب ویژگی‌ها
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
            'sma_20', 'ema_12', 'atr',
            'price_change', 'volume_change', 'high_low_ratio'
        ]
        
        # فقط ستون‌های موجود رو انتخاب کن
        available_cols = [col for col in feature_cols if col in df.columns]
        self.feature_columns = available_cols
        
        X = df[available_cols].values
        y = df['target'].values
        
        # نرمال‌سازی
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train_model(self, symbol: str, test_size: float = 0.2) -> Dict:
        """آموزش مدل روی داده‌های واقعی"""
        try:
            # دریافت داده‌های آموزشی
            df = market_analyzer.prepare_ai_training_data(symbol)
            
            if df.empty or len(df) < 100:
                logger.error(f"❌ داده‌های کافی برای آموزش {symbol} وجود ندارد")
                return {}
            
            # آماده‌سازی ویژگی‌ها
            X, y = self.prepare_features(df)
            
            # تقسیم داده
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=False  # مهم: shuffle=False برای داده زمانی
            )
            
            # تبدیل به تانسور
            X_train_tensor = torch.FloatTensor(X_train)
            X_test_tensor = torch.FloatTensor(X_test)
            y_train_tensor = torch.LongTensor(y_train + 1)  # تبدیل به 0,1,2
            y_test_tensor = torch.LongTensor(y_test + 1)
            
            # ایجاد مدل
            self.model = RealTradingModel(
                input_dim=X_train.shape[1],
                hidden_dims=[256, 128, 64, 32],
                dropout=0.2
            )
            
            # آموزش
            train_results = self._train_epochs(
                X_train_tensor, y_train_tensor,
                X_test_tensor, y_test_tensor
            )
            
            # ذخیره مدل
            self._save_model(symbol, train_results)
            
            logger.info(f"✅ مدل {symbol} آموزش داده شد")
            return train_results
            
        except Exception as e:
            logger.error(f"❌ خطا در آموزش مدل {symbol}: {e}")
            return {}
    
    def _train_epochs(self, X_train: torch.Tensor, y_train: torch.Tensor,
                     X_test: torch.Tensor, y_test: torch.Tensor,
                     epochs: int = 100, patience: int = 10) -> Dict:
        """آموزش مدل با Early Stopping"""
        
        optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        
        best_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # آموزش
            self.model.train()
            optimizer.zero_grad()
            
            outputs = self.model(X_train)
            loss = criterion(outputs['signals'], y_train)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Gradient Clipping
            optimizer.step()
            
            # ارزیابی
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_test)
                val_loss = criterion(val_outputs['signals'], y_test)
            
            train_losses.append(loss.item())
            val_losses.append(val_loss.item())
            
            # Early Stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"⏹️ توقف زودهنگام در دوره {epoch}")
                break
            
            if epoch % 20 == 0:
                logger.info(f"📊 دوره {epoch}: Loss={loss.item():.4f}, Val_Loss={val_loss.item():.4f}")
        
        # بارگذاری بهترین مدل
        self.model.load_state_dict(best_model_state)
        
        # محاسبه دقت نهایی
        with torch.no_grad():
            test_outputs = self.model(X_test)
            _, predicted = torch.max(test_outputs['signals'], 1)
            accuracy = (predicted == y_test).float().mean().item()
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_accuracy': accuracy,
            'best_val_loss': best_loss,
            'epochs_trained': epoch + 1
        }
    
    def _save_model(self, symbol: str, results: Dict):
        """ذخیره مدل آموزش دیده"""
        model_path = f"./models/{symbol}_model.pth"
        scaler_path = f"./models/{symbol}_scaler.pkl"
        
        # ایجاد پوشه مدل‌ها
        import os
        os.makedirs("./models", exist_ok=True)
        
        # ذخیره مدل
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'feature_columns': self.feature_columns,
            'training_results': results,
            'trained_at': datetime.now().isoformat()
        }, model_path)
        
        # ذخیره Scaler
        joblib.dump(self.scaler, scaler_path)
        
        logger.info(f"💾 مدل {symbol} ذخیره شد")
    
    def predict_real_time(self, symbol: str, current_data: Dict) -> Dict:
        """پیش‌بینی سیگنال در زمان واقعی"""
        try:
            if self.model is None:
                self._load_model(symbol)
            
            # آماده‌سازی داده ورودی
            features = self._prepare_real_time_features(current_data)
            
            # پیش‌بینی
            self.model.eval()
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).unsqueeze(0)
                outputs = self.model(features_tensor)
            
            # تفسیر نتایج
            signal_probs = torch.softmax(outputs['signals'][0], dim=0)
            confidence = outputs['confidence'].item()
            
            signal_types = ['SELL', 'HOLD', 'BUY']  # مطابق با targetهای -1, 0, 1
            best_signal_idx = torch.argmax(signal_probs).item()
            
            return {
                'symbol': symbol,
                'signal': signal_types[best_signal_idx],
                'confidence': signal_probs[best_signal_idx].item(),
                'model_confidence': confidence,
                'all_probabilities': {
                    sig: prob.item() for sig, prob in zip(signal_types, signal_probs)
                },
                'timestamp': datetime.now().isoformat(),
                'model_version': 'real_trained_v1'
            }
            
        except Exception as e:
            logger.error(f"❌ خطا در پیش‌بینی {symbol}: {e}")
            return {
                'symbol': symbol,
                'signal': 'HOLD',
                'confidence': 0.5,
                'model_confidence': 0.5,
                'error': str(e)
            }
    
    def _load_model(self, symbol: str):
        """بارگذاری مدل آموزش دیده"""
        model_path = f"./models/{symbol}_model.pth"
        scaler_path = f"./models/{symbol}_scaler.pkl"
        
        try:
            checkpoint = torch.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.feature_columns = checkpoint['feature_columns']
            
            self.model = RealTradingModel(input_dim=len(self.feature_columns))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            logger.info(f"📥 مدل {symbol} بارگذاری شد")
        except Exception as e:
            logger.error(f"❌ خطا در بارگذاری مدل {symbol}: {e}")
            raise
    
    def _prepare_real_time_features(self, data: Dict) -> np.ndarray:
        """آماده‌سازی ویژگی‌های زمان واقعی"""
        # استخراج ویژگی‌ها از داده ورودی
        features = []
        for col in self.feature_columns:
            features.append(data.get(col, 0.0))
        
        # نرمال‌سازی
        features_array = np.array(features).reshape(1, -1)
        return self.scaler.transform(features_array)

# ایجاد نمونه گلوبال
model_trainer = RealModelTrainer()
