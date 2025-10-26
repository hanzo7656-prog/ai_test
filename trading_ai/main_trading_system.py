import logging
from datetime import datetime

from database_manager import trading_db
from real_time_analyzer import market_analyzer
from model_trainer import model_trainer
from advanced_technical_engine import technical_engine
from config import trading_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalAnalysisSystem:
    """سیستم اصلی تحلیل تکنیکال هوشمند"""
    
    def __init__(self):
        self.is_trained = False
        self.analyzer = None
    
    def initialize_system(self):
        """راه‌اندازی سیستم تحلیل تکنیکال"""
        try:
            logger.info("🚀 راه‌اندازی سیستم تحلیل تکنیکال...")
            
            # بارگذاری و پیش‌پردازش داده‌ها
            for symbol in trading_config.SYMBOLS:
                logger.info(f"📥 آماده‌سازی داده‌های {symbol}...")
                market_analyzer.fetch_real_market_data(symbol, "1y")
            
            # آموزش مدل اسپارس
            logger.info("🤖 آموزش مدل اسپارس تحلیل تکنیکال...")
            training_results = model_trainer.train_technical_analysis(
                trading_config.SYMBOLS,
                epochs=trading_config.TRAINING_EPOCHS
            )
            
            if training_results:
                logger.info(f"📈 آموزش تکمیل شد - دقت: {training_results['final_accuracy']:.3f}")
                self.is_trained = True
                self.analyzer = model_trainer.model
            else:
                logger.error("❌ آموزش مدل با شکست مواجه شد")
            
            logger.info("✅ سیستم تحلیل تکنیکال راه‌اندازی شد")
            
        except Exception as e:
            logger.error(f"❌ خطا در راه‌اندازی سیستم: {e}")
    
    def analyze_symbol(self, symbol: str) -> Dict:
        """آنالیز تکنیکال یک نماد"""
        if not self.is_trained or self.analyzer is None:
            self.initialize_system()
        
        try:
            # دریافت داده‌های اخیر
            df = trading_db.get_historical_data(symbol, 100)
            if df.empty:
                return {'error': 'داده کافی موجود نیست'}
            
            # محاسبه اندیکاتورها
            df = technical_engine.calculate_all_indicators(df)
            
            # ایجاد دنباله ورودی برای مدل
            sequences, _ = technical_engine.create_sequences(df)
            if sequences is None or len(sequences) == 0:
                return {'error': 'دنباله زمانی کافی نیست'}
            
            # آخرین دنباله برای تحلیل
            latest_sequence = sequences[-1:]
            input_tensor = torch.FloatTensor(latest_sequence)
            
            # تحلیل با مدل اسپارس
            analysis = self.analyzer(input_tensor)
            
            # تفسیر نتایج
            interpretation = self.interpret_analysis(analysis, symbol)
            
            return interpretation
            
        except Exception as e:
            logger.error(f"❌ خطا در تحلیل {symbol}: {e}")
            return {'error': str(e)}
    
    def interpret_analysis(self, analysis: Dict, symbol: str) -> Dict:
        """تفسیر نتایج تحلیل مدل"""
        
        # تفسیر روند
        trend_probs = torch.softmax(analysis['trend_strength'][0], dim=-1)
        trend_labels = ['صعودی قوی', 'صعودی ضعیف', 'نزولی قوی', 'نزولی ضعیف', 'خنثی']
        trend_idx = torch.argmax(trend_probs).item()
        
        # تفسیر الگوها
        pattern_probs = torch.softmax(analysis['pattern_signals'][0], dim=-1)
        pattern_labels = ['سقف دوقلو', 'کف دوقلو', 'سر و شانه', 'مثلث', 'پرچم', 'کنج']
        pattern_idx = torch.argmax(pattern_probs).item()
        
        # تفسیر سطوح کلیدی
        levels = analysis['key_levels'][0].detach().numpy()
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'trend': {
                'direction': trend_labels[trend_idx],
                'confidence': trend_probs[trend_idx].item(),
                'all_probabilities': {label: prob.item() for label, prob in zip(trend_labels, trend_probs)}
            },
            'pattern': {
                'type': pattern_labels[pattern_idx],
                'confidence': pattern_probs[pattern_idx].item()
            },
            'key_levels': {
                'support': float(levels[0]),
                'resistance': float(levels[1]),
                'breakout': float(levels[2]),
                'breakdown': float(levels[3])
            },
            'volatility': float(analysis['market_volatility'][0].item()),
            'overall_confidence': float(analysis['overall_confidence'][0].item()),
            'specialty_activities': {
                specialty: float(activity[0].item()) 
                for specialty, activity in analysis['specialty_activities'].items()
            }
        }

# ایجاد نمونه سیستم
technical_system = TechnicalAnalysisSystem()

if __name__ == "__main__":
    # تست سیستم
    system = TechnicalAnalysisSystem()
    system.initialize_system()
    
    # تحلیل نمونه
    result = system.analyze_symbol('bitcoin')
    print("🎯 نتایج تحلیل تکنیکال:")
    print(f"نماد: {result['symbol']}")
    print(f"روند: {result['trend']['direction']} (اطمینان: {result['trend']['confidence']:.2f})")
    print(f"الگو: {result['pattern']['type']} (اطمینان: {result['pattern']['confidence']:.2f})")
    print(f"اطمینان کلی: {result['overall_confidence']:.2f}")
