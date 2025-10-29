# model_trainer.py - آموزش دهنده مدل اسپارس با داده‌های خام

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Optional
import logging
import joblib
from datetime import datetime
from trading_ai.database_manager import trading_db
from trading_ai.advanced_technical_engine import technical_engine
from trading_ai.sparse_technical_analyzer import SparseTechnicalNetwork, SparseConfig

logger = logging.getLogger(__name__)

class SparseModelTrainer:
    """آموزش دهنده مدل اسپارس تحلیل تکنیکال با داده‌های خام"""
    
    def __init__(self):
        self.config = SparseConfig()
        self.model = None
        self.scaler = None
        self.training_history = []
        logger.info("🚀 Sparse Model Trainer Initialized - Raw Data Mode")

    def train_technical_analysis(self, symbols: List[str], epochs: int = 30) -> Dict:
        """آموزش مدل اسپارس روی نمادهای مختلف با داده‌های خام"""
        try:
            all_sequences = []
            all_labels = []
            raw_data_quality = {}

            # جمع آوری داده های آموزشی از تمام نمادها
            for symbol in symbols:
                sequences, labels = technical_engine.prepare_training_data(symbol)
                
                if sequences is not None:
                    all_sequences.append(sequences)
                    all_labels.append(labels)
                    
                    # ثبت کیفیت داده‌های خام
                    raw_data_quality[symbol] = {
                        'sequences_count': len(sequences),
                        'data_points': sequences.shape[0] * sequences.shape[1] if len(sequences.shape) > 1 else 0,
                        'quality_score': self._calculate_sequence_quality(sequences)
                    }
                    
                    logger.info(f"✅ داده‌های {symbol} اضافه شد: {len(sequences)} نمونه")

            if not all_sequences:
                logger.error("❌ هیچ داده آموزشی یافت نشد")
                return {}

            # ترکیب داده های تمام نمادها
            X_train = np.concatenate(all_sequences, axis=0)
            y_train = np.concatenate(all_labels, axis=0)

            logger.info(f"📊 داده‌های آموزشی ترکیب شد: {X_train.shape}")

            # ایجاد مدل اسپارس
            self.model = SparseTechnicalNetwork(self.config)

            # آموزش مدل
            training_results = self._train_model(X_train, y_train, epochs)

            # ذخیره مدل آموزش دیده
            self._save_model(symbols[0], training_results, raw_data_quality)

            logger.info("✅ آموزش مدل اسپارس تکمیل شد")
            return training_results

        except Exception as e:
            logger.error(f"❌ خطا در آموزش مدل: {e}")
            return {}

    def _calculate_sequence_quality(self, sequences: np.ndarray) -> float:
        """محاسبه کیفیت دنباله‌های داده خام"""
        try:
            if sequences.size == 0:
                return 0.0
            
            # بررسی وجود مقادیر نامعتبر
            invalid_count = np.isnan(sequences).sum() + np.isinf(sequences).sum()
            validity_score = 1.0 - (invalid_count / sequences.size)
            
            # بررسی تنوع داده‌ها
            variance_score = min(np.var(sequences) / 10, 1.0)  # نرمال‌سازی واریانس
            
            # نمره کیفیت ترکیبی
            quality_score = (validity_score * 0.7) + (variance_score * 0.3)
            return round(quality_score, 3)
            
        except Exception as e:
            logger.error(f"❌ خطا در محاسبه کیفیت دنباله: {e}")
            return 0.0

    def _train_model(self, X: np.ndarray, y: np.ndarray, epochs: int) -> Dict:
        """آموزش مدل با داده های آماده شده"""
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
        best_model_state = None

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
                logger.info(f"📈 دوره {epoch}: Train_Loss={train_losses[-1]:.4f}, Val_Loss={val_losses[-1]:.4f}")

        # بارگذاری بهترین مدل
        if best_model_state:
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
            'epochs_trained': epoch + 1,
            'training_samples': len(X_train),
            'validation_samples': len(X_val)
        }

    def _save_model(self, symbol: str, results: Dict, raw_data_quality: Dict):
        """ذخیره مدل آموزش دیده با metadata داده‌های خام"""
        import os
        os.makedirs("./models", exist_ok=True)

        model_path = f"./models/sparse_technical_{symbol}.pth"

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'training_results': results,
            'config': self.config.__dict__,
            'trained_at': datetime.now().isoformat(),
            'model_type': 'sparse_technical_analyzer',
            'raw_data_quality': raw_data_quality,
            'data_sources': ['CoinStats', 'WebSocket', 'Historical'],
            'input_features': self.config.input_features,
            'sequence_length': self.config.sequence_length
        }, model_path)

        logger.info(f"💾 مدل اسپارس ذخیره شد: {model_path}")

    def load_model(self, model_path: str):
        """بارگذاری مدل آموزش دیده"""
        checkpoint = torch.load(model_path)

        self.config = SparseConfig(**checkpoint['config'])
        self.model = SparseTechnicalNetwork(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # نمایش اطلاعات داده‌های خام استفاده شده
        raw_data_quality = checkpoint.get('raw_data_quality', {})
        if raw_data_quality:
            logger.info("📊 اطلاعات داده‌های خام مدل:")
            for symbol, quality in raw_data_quality.items():
                logger.info(f"  {symbol}: {quality['sequences_count']} نمونه - کیفیت: {quality['quality_score']}")

        logger.info(f"✅ مدل اسپارس بارگذاری شد: {model_path}")

    def evaluate_model(self, test_symbols: List[str]) -> Dict[str, Any]:
        """ارزیابی مدل روی نمادهای تست"""
        try:
            evaluation_results = {}
            
            for symbol in test_symbols:
                # آماده‌سازی داده‌های تست
                sequences, labels = technical_engine.prepare_training_data(symbol)
                
                if sequences is None or len(sequences) == 0:
                    logger.warning(f"⚠️ داده‌های تست برای {symbol} موجود نیست")
                    continue
                
                # تبدیل به تانسور
                X_test = torch.FloatTensor(sequences)
                y_test = torch.LongTensor(labels)
                
                # ارزیابی مدل
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(X_test)
                    _, predicted = torch.max(outputs['trend_strength'], 1)
                    accuracy = (predicted == y_test).float().mean().item()
                    
                    # محاسبه سایر متریک‌ها
                    confidence_scores = torch.softmax(outputs['trend_strength'], dim=1).max(dim=1).values
                    avg_confidence = confidence_scores.mean().item()
                
                evaluation_results[symbol] = {
                    'accuracy': accuracy,
                    'avg_confidence': avg_confidence,
                    'test_samples': len(sequences),
                    'prediction_distribution': torch.bincount(predicted).tolist()
                }
                
                logger.info(f"📊 ارزیابی {symbol}: دقت={accuracy:.3f}, اطمینان={avg_confidence:.3f}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"❌ خطا در ارزیابی مدل: {e}")
            return {}

    def get_training_status(self) -> Dict[str, Any]:
        """دریافت وضعیت فعلی آموزش"""
        status = {
            'model_loaded': self.model is not None,
            'config': self.config.__dict__ if hasattr(self, 'config') else {},
            'last_training': self.training_history[-1] if self.training_history else None,
            'raw_data_mode': True,
            'available_features': self.config.input_features,
            'sequence_length': self.config.sequence_length
        }
        
        if self.model is not None:
            status['model_parameters'] = sum(p.numel() for p in self.model.parameters())
            status['model_specialties'] = self.config.specialty_groups
            
        return status

    def fine_tune_model(self, new_symbols: List[str], fine_tune_epochs: int = 10) -> Dict:
        """Fine-tuning مدل روی نمادهای جدید"""
        try:
            if self.model is None:
                logger.error("❌ هیچ مدلی برای Fine-tuning بارگذاری نشده")
                return {}
            
            logger.info(f"🔧 Fine-tuning مدل روی {len(new_symbols)} نماد جدید")
            
            # جمع‌آوری داده‌های جدید
            new_sequences = []
            new_labels = []
            
            for symbol in new_symbols:
                sequences, labels = technical_engine.prepare_training_data(symbol)
                if sequences is not None:
                    new_sequences.append(sequences)
                    new_labels.append(labels)
                    logger.info(f"✅ داده‌های {symbol} برای Fine-tuning اضافه شد")
            
            if not new_sequences:
                logger.error("❌ هیچ داده جدیدی برای Fine-tuning یافت نشد")
                return {}
            
            # ترکیب داده‌های جدید
            X_new = np.concatenate(new_sequences, axis=0)
            y_new = np.concatenate(new_labels, axis=0)
            
            # Fine-tuning با نرخ یادگیری پایین‌تر
            optimizer = optim.AdamW(self.model.parameters(), lr=0.0001, weight_decay=0.001)
            criterion = nn.CrossEntropyLoss()
            
            X_tensor = torch.FloatTensor(X_new)
            y_tensor = torch.LongTensor(y_new)
            
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
            
            fine_tune_losses = []
            
            for epoch in range(fine_tune_epochs):
                self.model.train()
                epoch_loss = 0
                
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs['trend_strength'], batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    optimizer.step()
                    epoch_loss += loss.item()
                
                fine_tune_losses.append(epoch_loss / len(dataloader))
                
                if epoch % 2 == 0:
                    logger.info(f"🔧 Fine-tuning دوره {epoch}: Loss={fine_tune_losses[-1]:.4f}")
            
            # ذخیره مدل Fine-tuned شده
            self._save_model("fine_tuned", {
                'fine_tune_losses': fine_tune_losses,
                'fine_tune_epochs': fine_tune_epochs,
                'fine_tune_symbols': new_symbols
            }, {})
            
            return {
                'fine_tune_losses': fine_tune_losses,
                'final_loss': fine_tune_losses[-1],
                'trained_symbols': new_symbols,
                'total_samples': len(X_new)
            }
            
        except Exception as e:
            logger.error(f"❌ خطا در Fine-tuning مدل: {e}")
            return {}

# ایجاد نمونه گلوبال
model_trainer = SparseModelTrainer()
