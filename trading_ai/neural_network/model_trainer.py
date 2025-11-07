# آموزش‌دهنده مدل‌های شبکه عصبی
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class ModelTrainer:
    """آموزش‌دهنده مدل‌های شبکه عصبی برای داده‌های مالی"""
    
    def __init__(self, neural_network, config=None):
        self.neural_network = neural_network
        self.config = config or {}
        self.training_data = []
        self.validation_data = []
        logger.info("✅ Model Trainer initialized")
    
    def prepare_training_data(self, market_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """آماده‌سازی داده‌های آموزش"""
        try:
            features = []
            labels = []
            
            for data in market_data:
                # استخراج ویژگی‌ها
                feature_vector = self._extract_features(data)
                features.append(feature_vector)
                
                # استخراج برچسب‌ها
                label_vector = self._create_label(data)
                labels.append(label_vector)
            
            return np.array(features), np.array(labels)
            
        except Exception as e:
            logger.error(f"خطا در آماده‌سازی داده‌های آموزش: {e}")
            return np.array([]), np.array([])
    
    def _extract_features(self, market_data: Dict[str, Any]) -> List[float]:
        """استخراج ویژگی‌ها از داده‌های بازار"""
        try:
            features = []
            
            # ویژگی‌های اصلی
            price = market_data.get('price', 0)
            price_change = market_data.get('priceChange1d', 0)
            volume = market_data.get('volume', 0)
            market_cap = market_data.get('marketCap', 0)
            rank = market_data.get('rank', 100)
            
            # نرمال‌سازی و اضافه کردن
            features.extend([
                price / 100000,
                price_change / 100,
                np.log(volume + 1) / 20,
                np.log(market_cap + 1) / 25,
                rank / 100
            ])
            
            # ویژگی‌های تکنیکال
            if 'historical_prices' in market_data:
                prices = market_data['historical_prices']
                if len(prices) >= 20:
                    # نوسان
                    returns = np.diff(prices) / prices[:-1]
                    volatility = np.std(returns) * 100 if len(returns) > 0 else 0
                    features.append(volatility / 50)
                    
                    # روند
                    if len(prices) >= 50:
                        trend_slope = self._calculate_trend_slope(prices[-50:])
                        features.append(trend_slope)
                    else:
                        features.append(0.0)
                else:
                    features.extend([0.0, 0.0])
            else:
                features.extend([0.0, 0.0])
            
            # پر کردن تا ۲۰ ویژگی
            while len(features) < 20:
                features.append(0.0)
            
            return features[:20]
            
        except Exception as e:
            logger.error(f"خطا در استخراج ویژگی‌ها: {e}")
            return [0.0] * 20
    
    def _calculate_trend_slope(self, prices: List[float]) -> float:
        """محاسبه شیب روند"""
        try:
            x = np.arange(len(prices))
            slope, _ = np.polyfit(x, prices, 1)
            return float(slope / (np.std(prices) + 1e-8))
        except:
            return 0.0
    
    def _create_label(self, market_data: Dict[str, Any]) -> List[float]:
        """ایجاد برچسب برای آموزش"""
        try:
            # برچسب‌های one-hot encoding
            # [STRONG_SELL, SELL, HOLD, BUY, STRONG_BUY]
            price_change = market_data.get('priceChange1d', 0)
            volume_change = market_data.get('volumeChange', 0)
            
            # منطق ساده برای برچسب‌گذاری
            if price_change > 10 and volume_change > 20:
                return [0, 0, 0, 0, 1]  # STRONG_BUY
            elif price_change > 5:
                return [0, 0, 0, 1, 0]  # BUY
            elif price_change < -10 and volume_change > 20:
                return [1, 0, 0, 0, 0]  # STRONG_SELL
            elif price_change < -5:
                return [0, 1, 0, 0, 0]  # SELL
            else:
                return [0, 0, 1, 0, 0]  # HOLD
                
        except Exception as e:
            logger.error(f"خطا در ایجاد برچسب: {e}")
            return [0, 0, 1, 0, 0]  # HOLD به عنوان پیش‌فرض
    
    def train_model(self, training_data: List[Dict[str, Any]], 
                   validation_data: List[Dict[str, Any]] = None,
                   epochs: int = 100,
                   learning_rate: float = 0.01) -> Dict[str, Any]:
        """آموزش مدل شبکه عصبی"""
        try:
            logger.info(f"🚀 شروع آموزش مدل با {len(training_data)} نمونه")
            
            # آماده‌سازی داده‌ها
            X_train, y_train = self.prepare_training_data(training_data)
            
            if X_train.size == 0:
                raise ValueError("داده‌های آموزش خالی هستند")
            
            # آموزش مدل
            self.neural_network.train(X_train, y_train, epochs=epochs, learning_rate=learning_rate)
            
            # ارزیابی مدل
            training_results = self.evaluate_model(training_data, "آموزش")
            validation_results = {}
            
            if validation_data:
                validation_results = self.evaluate_model(validation_data, "اعتبارسنجی")
            
            results = {
                'training_samples': len(training_data),
                'validation_samples': len(validation_data) if validation_data else 0,
                'training_accuracy': training_results.get('accuracy', 0),
                'training_loss': training_results.get('loss', 0),
                'validation_accuracy': validation_results.get('accuracy', 0) if validation_results else 0,
                'validation_loss': validation_results.get('loss', 0) if validation_results else 0,
                'epochs_trained': epochs,
                'learning_rate': learning_rate,
                'completion_time': datetime.now().isoformat()
            }
            
            logger.info(f"✅ آموزش مدل کامل شد - دقت: {results['training_accuracy']:.2f}")
            return results
            
        except Exception as e:
            logger.error(f"❌ خطا در آموزش مدل: {e}")
            return {'error': str(e), 'success': False}
    
    def evaluate_model(self, test_data: List[Dict[str, Any]], dataset_name: str = "تست") -> Dict[str, Any]:
        """ارزیابی مدل آموزش دیده"""
        try:
            if not self.neural_network.is_trained:
                return {'accuracy': 0, 'loss': 0, 'error': 'مدل آموزش ندیده'}
            
            X_test, y_test = self.prepare_training_data(test_data)
            
            if X_test.size == 0:
                return {'accuracy': 0, 'loss': 0, 'error': 'داده‌های تست خالی'}
            
            # پیش‌بینی
            predictions = self.neural_network.forward(X_test)
            
            # محاسبه دقت
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(y_test, axis=1)
            accuracy = np.mean(predicted_classes == true_classes)
            
            # محاسبه خطا
            loss = -np.sum(y_test * np.log(predictions + 1e-8)) / len(y_test)
            
            # ماتریس درهم‌ریختگی
            confusion_matrix = self._compute_confusion_matrix(predicted_classes, true_classes)
            
            results = {
                'accuracy': float(accuracy),
                'loss': float(loss),
                'dataset': dataset_name,
                'samples': len(test_data),
                'confusion_matrix': confusion_matrix,
                'class_distribution': self._get_class_distribution(true_classes)
            }
            
            logger.info(f"📊 ارزیابی {dataset_name}: دقت={accuracy:.2f}, خطا={loss:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"خطا در ارزیابی مدل: {e}")
            return {'accuracy': 0, 'loss': 0, 'error': str(e)}
    
    def _compute_confusion_matrix(self, predictions: np.ndarray, true_labels: np.ndarray) -> Dict[str, Any]:
        """محاسبه ماتریس درهم‌ریختگی"""
        try:
            classes = ['STRONG_SELL', 'SELL', 'HOLD', 'BUY', 'STRONG_BUY']
            matrix = {}
            
            for i, true_class in enumerate(classes):
                matrix[true_class] = {}
                for j, pred_class in enumerate(classes):
                    count = np.sum((true_labels == i) & (predictions == j))
                    matrix[true_class][pred_class] = int(count)
            
            return matrix
            
        except Exception as e:
            logger.error(f"خطا در محاسبه ماتریس درهم‌ریختگی: {e}")
            return {}
    
    def _get_class_distribution(self, labels: np.ndarray) -> Dict[str, int]:
        """توزیع کلاس‌ها در داده‌ها"""
        try:
            classes = ['STRONG_SELL', 'SELL', 'HOLD', 'BUY', 'STRONG_BUY']
            distribution = {}
            
            for i, class_name in enumerate(classes):
                count = np.sum(labels == i)
                distribution[class_name] = int(count)
            
            return distribution
            
        except Exception as e:
            logger.error(f"خطا در محاسبه توزیع کلاس‌ها: {e}")
            return {}
    
    def save_training_report(self, results: Dict[str, Any], filepath: str = "training_report.json"):
        """ذخیره گزارش آموزش"""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            report = {
                'training_results': results,
                'network_info': self.neural_network.get_network_info(),
                'training_date': datetime.now().isoformat(),
                'config': self.config
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 گزارش آموزش در {filepath} ذخیره شد")
            
        except Exception as e:
            logger.error(f"خطا در ذخیره گزارش آموزش: {e}")
    
    def load_training_data(self, filepath: str) -> List[Dict[str, Any]]:
        """بارگذاری داده‌های آموزش از فایل"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"📂 داده‌های آموزش از {filepath} بارگذاری شد")
            return data
            
        except Exception as e:
            logger.error(f"خطا در بارگذاری داده‌های آموزش: {e}")
            return []
