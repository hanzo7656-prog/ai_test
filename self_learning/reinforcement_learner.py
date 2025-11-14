# self_learning/reinforcement_learner.py
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import random
import math

logger = logging.getLogger(__name__)

# ساختار برای ذخیره تجربیات
Experience = namedtuple('Experience', 
                       ['state', 'action', 'reward', 'next_state', 'done', 'timestamp'])

class ReinforcementNetwork(nn.Module):
    """شبکه عصبی برای یادگیری تقویتی"""
    
    def __init__(self, state_size: int, action_size: int, hidden_layers: List[int] = [128, 64]):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        layers = []
        input_size = state_size
        
        # ایجاد لایه‌های پنهان
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_size = hidden_size
        
        # لایه خروجی
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class ReinforcementLearner:
    """سیستم یادگیری تقویتی پیشرفته برای بهینه‌سازی مدل‌ها"""
    
    def __init__(self, model_manager, state_size: int = 50, action_size: int = 10):
        self.model_manager = model_manager
        self.state_size = state_size
        self.action_size = action_size
        
        # شبکه‌های اصلی و target
        self.q_network = ReinforcementNetwork(state_size, action_size)
        self.target_network = ReinforcementNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        
        # replay buffer برای تجربیات
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # پارامترهای یادگیری
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.tau = 0.01  # برای soft update
        
        # تاریخچه یادگیری
        self.learning_history = []
        self.episode_rewards = []
        
        # ارتباط با سیستم کش
        from debug_system.storage.cache_debugger import cache_debugger
        self.cache_manager = cache_debugger
        
        logger.info("🤖 Reinforcement Learner initialized")

    def add_experience(self, state: np.ndarray, action: int, reward: float, 
                      next_state: np.ndarray, done: bool):
        """ذخیره تجربه جدید در replay buffer"""
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            timestamp=datetime.now().isoformat()
        )
        self.memory.append(experience)
        
        # یادگیری از تجربیات اگر به اندازه کافی نمونه داریم
        if len(self.memory) > self.batch_size:
            self._learn_from_experiences()

    def _learn_from_experiences(self):
        """یادگیری از نمونه‌های replay buffer"""
        try:
            # نمونه‌گیری تصادفی از replay buffer
            batch = random.sample(self.memory, self.batch_size)
            
            states = torch.FloatTensor([exp.state for exp in batch])
            actions = torch.LongTensor([exp.action for exp in batch])
            rewards = torch.FloatTensor([exp.reward for exp in batch])
            next_states = torch.FloatTensor([exp.next_state for exp in batch])
            dones = torch.BoolTensor([exp.done for exp in batch])
            
            # محاسبه Q-values فعلی
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            
            # محاسبه target Q-values
            with torch.no_grad():
                next_q_values = self.target_network(next_states).max(1)[0]
                target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            
            # محاسبه loss
            loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
            
            # بهینه‌سازی
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # کاهش exploration rate
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # soft update target network
            self._soft_update_target_network()
            
            # ذخیره تاریخچه یادگیری
            learning_step = {
                'timestamp': datetime.now().isoformat(),
                'loss': loss.item(),
                'epsilon': self.epsilon,
                'memory_size': len(self.memory),
                'average_reward': np.mean([exp.reward for exp in batch])
            }
            self.learning_history.append(learning_step)
            
            # ذخیره در کش هر ۱۰۰ step
            if len(self.learning_history) % 100 == 0:
                self._save_learning_progress()
                
        except Exception as e:
            logger.error(f"❌ Error in reinforcement learning: {e}")

    def _soft_update_target_network(self):
        """Soft update برای target network"""
        for target_param, local_param in zip(self.target_network.parameters(), 
                                           self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + 
                                  (1.0 - self.tau) * target_param.data)

    def get_action(self, state: np.ndarray) -> int:
        """دریافت action بر اساس state فعلی"""
        if np.random.random() < self.epsilon:
            # Exploration: action تصادفی
            return random.randint(0, self.action_size - 1)
        else:
            # Exploitation: بهترین action بر اساس Q-values
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()

    def optimize_model_parameters(self, model_name: str, state: np.ndarray) -> Dict[str, Any]:
        """بهینه‌سازی پارامترهای مدل با یادگیری تقویتی"""
        try:
            if model_name not in self.model_manager.active_models:
                raise ValueError(f"Model {model_name} not found")
            
            # دریافت action بهینه
            action = self.get_action(state)
            
            # اعمال action به مدل (مثلاً تنظیم learning rate، تغییر architecture)
            reward = self._apply_action_to_model(model_name, action)
            
            # مشاهده state بعدی
            next_state = self._get_next_state(model_name, state, action)
            
            # ذخیره تجربه
            self.add_experience(state, action, reward, next_state, done=False)
            
            optimization_result = {
                'model': model_name,
                'action_taken': action,
                'reward_earned': reward,
                'epsilon': self.epsilon,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"🎯 RL optimization for {model_name}: action={action}, reward={reward:.3f}")
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"❌ RL optimization failed for {model_name}: {e}")
            return {'error': str(e)}

    def _apply_action_to_model(self, model_name: str, action: int) -> float:
        """اعمال action به مدل و دریافت reward"""
        try:
            model_info = self.model_manager.active_models[model_name]
            model = model_info['model']
            
            # بر اساس action، تغییرات مختلف اعمال می‌شود
            if action == 0:
                # افزایش learning rate
                reward = self._adjust_learning_rate(model, 1.1)
            elif action == 1:
                # کاهش learning rate
                reward = self._adjust_learning_rate(model, 0.9)
            elif action == 2:
                # افزایش dropout
                reward = self._adjust_dropout(model, 1.1)
            elif action == 3:
                # کاهش dropout
                reward = self._adjust_dropout(model, 0.9)
            else:
                # سایر تنظیمات
                reward = self._other_adjustments(model, action)
            
            return reward
            
        except Exception as e:
            logger.error(f"❌ Error applying action to model: {e}")
            return -1.0  # reward منفی برای خطا

    def _adjust_learning_rate(self, model, factor: float) -> float:
        """تنظیم learning rate و ارزیابی reward"""
        # این یک پیاده‌سازی ساده است
        # در عمل باید بر اساس performance مدل reward محاسبه شود
        return 0.1  # placeholder

    def _adjust_dropout(self, model, factor: float) -> float:
        """تنظیم dropout rate"""
        return 0.1  # placeholder

    def _other_adjustments(self, model, action: int) -> float:
        """سایر تنظیمات مدل"""
        return 0.05  # placeholder

    def _get_next_state(self, model_name: str, current_state: np.ndarray, action: int) -> np.ndarray:
        """دریافت state بعدی پس از اعمال action"""
        # این تابع state جدید را پس از اعمال action برمی‌گرداند
        # می‌تواند شامل performance جدید مدل باشد
        return current_state  # placeholder

    def train_trading_agent(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """آموزش agent برای معامله‌گری"""
        try:
            # تبدیل داده بازار به state
            state = self._market_data_to_state(market_data)
            
            # دریافت action
            action = self.get_action(state)
            
            # شبیه‌سازی معامله و دریافت reward
            reward, done = self._simulate_trade(action, market_data)
            
            # state بعدی
            next_state = self._get_next_trading_state(state, action, market_data)
            
            # ذخیره تجربه
            self.add_experience(state, action, reward, next_state, done)
            
            training_result = {
                'episode': len(self.episode_rewards) + 1,
                'action': self._action_to_trade_type(action),
                'reward': reward,
                'epsilon': self.epsilon,
                'timestamp': datetime.now().isoformat()
            }
            
            if done:
                self.episode_rewards.append(reward)
                training_result['episode_complete'] = True
                training_result['total_episode_reward'] = reward
            
            return training_result
            
        except Exception as e:
            logger.error(f"❌ Trading agent training failed: {e}")
            return {'error': str(e)}

    def _market_data_to_state(self, market_data: Dict[str, Any]) -> np.ndarray:
        """تبدیل داده بازار به state برای RL"""
        # استخراج ویژگی‌های کلیدی از داده بازار
        features = []
        
        # قیمت و حجم
        if 'raw_coins' in market_data.get('sources', {}):
            coin_data = market_data['sources']['raw_coins'].get('data', [])
            if coin_data and isinstance(coin_data, list):
                prices = [item.get('price', 0) for item in coin_data[:10] if isinstance(item, dict)]
                if prices:
                    features.extend([
                        np.mean(prices),
                        np.std(prices),
                        (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
                    ])
        
        # احساسات بازار
        if 'raw_news' in market_data.get('sources', {}):
            news_data = market_data['sources']['raw_news'].get('data', [])
            if news_data and isinstance(news_data, list):
                # تحلیل ساده احساسات
                sentiment = self._calculate_news_sentiment(news_data)
                features.append(sentiment)
        
        # اگر ویژگی‌ها کم هستند، padding کن
        while len(features) < self.state_size:
            features.append(0.0)
        
        return np.array(features[:self.state_size], dtype=np.float32)

    def _calculate_news_sentiment(self, news_data: List[Dict]) -> float:
        """محاسبه احساسات از داده‌های خبری"""
        if not news_data:
            return 0.5
        
        positive_words = ['صعود', 'رشد', 'سود', 'مثبت', 'قوی']
        negative_words = ['نزول', 'سقوط', 'ضرر', 'منفی', 'ضعیف']
        
        total_sentiment = 0
        for news_item in news_data[:5]:  # فقط ۵ خبر اول
            text = f"{news_item.get('title', '')} {news_item.get('description', '')}".lower()
            positive_count = sum(1 for word in positive_words if word in text)
            negative_count = sum(1 for word in negative_words if word in text)
            
            total = positive_count + negative_count
            if total > 0:
                total_sentiment += positive_count / total
        
        return total_sentiment / min(5, len(news_data)) if news_data else 0.5

    def _action_to_trade_type(self, action: int) -> str:
        """تبدیل action به نوع معامله"""
        actions = {
            0: "BUY",
            1: "SELL", 
            2: "HOLD",
            3: "BUY_AGGRESSIVE",
            4: "SELL_AGGRESSIVE"
        }
        return actions.get(action, "HOLD")

    def _simulate_trade(self, action: int, market_data: Dict[str, Any]) -> Tuple[float, bool]:
        """شبیه‌سازی معامله و محاسبه reward"""
        # این یک شبیه‌سازی ساده است
        # در عمل باید با داده واقعی بازار کار کند
        
        reward = random.uniform(-1.0, 1.0)  # شبیه‌سازی ساده
        done = random.random() < 0.1  # 10% chance episode تمام شود
        
        return reward, done

    def _get_next_trading_state(self, current_state: np.ndarray, action: int, 
                              market_data: Dict[str, Any]) -> np.ndarray:
        """دریافت state بعدی برای معامله"""
        # در عمل باید state جدید بر اساس action و تغییرات بازار باشد
        return current_state  # placeholder

    def _save_learning_progress(self):
        """ذخیره پیشرفت یادگیری در کش"""
        try:
            progress_data = {
                'learning_history': self.learning_history[-100:],  # ۱۰۰ نمونه آخر
                'episode_rewards': self.episode_rewards,
                'epsilon': self.epsilon,
                'memory_size': len(self.memory),
                'timestamp': datetime.now().isoformat()
            }
            
            self.cache_manager.set_data("utb", "rl_learning_progress", progress_data, expire=3600)
            
        except Exception as e:
            logger.error(f"❌ Error saving RL progress: {e}")

    def get_learning_stats(self) -> Dict[str, Any]:
        """دریافت آمار یادگیری"""
        return {
            'timestamp': datetime.now().isoformat(),
            'memory_size': len(self.memory),
            'epsilon': self.epsilon,
            'total_episodes': len(self.episode_rewards),
            'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'learning_steps': len(self.learning_history),
            'recent_loss': self.learning_history[-1]['loss'] if self.learning_history else 0
        }

# نمونه global
reinforcement_learner = None

def initialize_reinforcement_learner(model_manager):
    """مقداردهی اولیه reinforcement learner"""
    global reinforcement_learner
    reinforcement_learner = ReinforcementLearner(model_manager)
    return reinforcement_learner
