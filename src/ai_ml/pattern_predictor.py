# 📁 src/ai_ml/pattern_predictor.py

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
import os

class LSTMPatternPredictor(nn.Module):
    """شبکه LSTM برای پیش‌بینی الگوهای قیمتی"""
    
    def __init__(self, input_size: int = 5, hidden_size: int = 50, 
                 num_layers: int = 2, output_size: int = 3):
        super(LSTMPatternPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 25),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(25, output_size),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # استفاده از خروجی آخرین لایه
        last_time_step = lstm_out[:, -1, :]
        
        # لایه fully connected
        output = self.fc(last_time_step)
        
        return output

class PatternPredictor:
    """پیش‌بین الگوهای قیمتی با LSTM"""
    
    def __init__(self, sequence_length: int = 20, model_path: str = "models/pattern_predictor.pth"):
        self.sequence_length = sequence_length
        self.model_path = model_path
        self.scaler = MinMaxScaler()
        self.model = None
        self.is_trained = False
        
        # الگوهای خروجی
        self.patterns = {
            0: 'UPTREND_CONTINUATION',   # ادامه روند صعودی
            1: 'DOWNTREND_CONTINUATION', # ادامه روند نزولی
            2: 'REVERSAL'                # بازگشت روند
        }
    
    def prepare_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """آماده‌سازی دنباله‌ها برای آموزش"""
        features = self._extract_features(data)
        
        # نرمال‌سازی
        features_scaled = self.scaler.fit_transform(features)
        
        X, y = [], []
        
        for i in range(self.sequence_length, len(features_scaled)):
            # دنباله ورودی
            X.append(features_scaled[i-self.sequence_length:i])
            
            # لیبل خروجی (تغییر قیمت در آینده)
            future_return = (
                data['close'].iloc[i+5] / data['close'].iloc[i] - 1 
                if i + 5 < len(data) else 0
            )
            
            if future_return > 0.02:
                y.append(0)  # UPTREND_CONTINUATION
            elif future_return < -0.02:
                y.append(1)  # DOWNTREND_CONTINUATION
            else:
                y.append(2)  # REVERSAL
        
        return np.array(X), np.array(y)
    
    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """استخراج ویژگی‌ها از داده قیمت"""
        features = pd.DataFrame(index=data.index)
        
        # ویژگی‌های قیمتی
        features['open'] = data['open']
        features['high'] = data['high']
        features['low'] = data['low']
        features['close'] = data['close']
        
        if 'volume' in data.columns:
            features['volume'] = data['volume']
        
        # ویژگی‌های تکنیکال
        features['returns'] = data['close'].pct_change()
        features['volatility'] = data['close'].pct_change().rolling(5).std()
        
        # میانگین‌های متحرک
        features['ma_5'] = data['close'].rolling(5).mean()
        features['ma_10'] = data['close'].rolling(10).mean()
        features['ma_20'] = data['close'].rolling(20).mean()
        
        # RSI ساده
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        return features.dropna()
    
    def train(self, data: pd.DataFrame, epochs: int = 100) -> Dict:
        """آموزش مدل LSTM"""
        print("🔄 Training Pattern Predictor LSTM...")
        
        # آماده‌سازی داده
        X, y = self.prepare_sequences(data)
        
        if len(X) < 100:
            print("❌ Insufficient sequences for training")
            return {}
        
        # تقسیم داده
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # تبدیل به تنسور
        X_train_tensor = torch.FloatTensor(X_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_train_tensor = torch.LongTensor(y_train)
        y_test_tensor = torch.LongTensor(y_test)
        
        # ایجاد مدل
        input_size = X_train.shape[2]
        self.model = LSTMPatternPredictor(
            input_size=input_size,
            hidden_size=50,
            num_layers=2,
            output_size=3
        )
        
        # آموزش
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        train_losses = []
        test_accuracies = []
        
        for epoch in range(epochs):
            # حالت آموزش
            self.model.train()
            
            # Forward pass
            outputs = self.model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # ارزیابی
            self.model.eval()
            with torch.no_grad():
                test_outputs = self.model(X_test_tensor)
                _, predicted = torch.max(test_outputs, 1)
                test_accuracy = (predicted == y_test_tensor).float().mean()
            
            train_losses.append(loss.item())
            test_accuracies.append(test_accuracy.item())
            
            if epoch % 20 == 0:
                print(f"   Epoch {epoch}: Loss = {loss.item():.4f}, Test Acc = {test_accuracy.item():.4f}")
        
        self.is_trained = True
        
        # ذخیره مدل
        self._save_model()
        
        final_accuracy = test_accuracies[-1]
        print(f"✅ LSTM Training Completed - Final Accuracy: {final_accuracy:.4f}")
        
        return {
            'final_accuracy': final_accuracy,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'input_features': input_size
        }
    
    def predict_pattern(self, data: pd.DataFrame) -> Dict:
        """پیش‌بینی الگوی آینده"""
        if not self.is_trained or self.model is None:
            if not self._load_model():
                return {'pattern': 'UNKNOWN', 'confidence': 0.0}
        
        # آماده‌سازی آخرین دنباله
        features = self._extract_features(data)
        if len(features) < self.sequence_length:
            return {'pattern': 'UNKNOWN', 'confidence': 0.0}
        
        latest_sequence = features.iloc[-self.sequence_length:]
        latest_sequence_scaled = self.scaler.transform(latest_sequence)
        
        # تبدیل به تنسور
        sequence_tensor = torch.FloatTensor(latest_sequence_scaled).unsqueeze(0)
        
        # پیش‌بینی
        self.model.eval()
        with torch.no_grad():
            output = self.model(sequence_tensor)
            probabilities = output[0].numpy()
            predicted_class = np.argmax(probabilities)
        
        confidence = probabilities[predicted_class]
        pattern = self.patterns.get(predicted_class, 'UNKNOWN')
        
        return {
            'pattern': pattern,
            'confidence': confidence,
            'all_probabilities': {
                self.patterns[i]: prob for i, prob in enumerate(probabilities)
            },
            'time_horizon': '5_periods'
        }
    
    def _save_model(self):
        """ذخیره مدل آموزش دیده"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'sequence_length': self.sequence_length
        }, self.model_path)
    
    def _load_model(self) -> bool:
        """لود مدل ذخیره شده"""
        try:
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path)
                
                input_size = 5  # تعداد ویژگی‌های پایه
                self.model = LSTMPatternPredictor(input_size=input_size)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.scaler = checkpoint['scaler']
                self.sequence_length = checkpoint['sequence_length']
                self.is_trained = True
                
                print("✅ LSTM Model loaded successfully")
                return True
        except Exception as e:
            print(f"❌ Error loading LSTM model: {e}")
        
        return False
