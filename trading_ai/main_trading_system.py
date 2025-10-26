# main_trading_system.py - سیستم اصلی تریدینگ واقعی
import logging
from datetime import datetime

from database_manager import trading_db
from real_time_analyzer import market_analyzer
from model_trainer import model_trainer
from backtest_engine import backtest_engine
from config import trading_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTradingSystem:
    """سیستم تریدینگ واقعی"""
    
    def __init__(self):
        self.is_initialized = False
    
    def initialize_system(self):
        """راه‌اندازی اولیه سیستم"""
        try:
            logger.info("🚀 راه‌اندازی سیستم تریدینگ واقعی...")
            
            # بارگذاری داده‌های تاریخی
            for symbol in trading_config.SYMBOLS:
                logger.info(f"📥 بارگذاری داده‌های {symbol}...")
                market_analyzer.fetch_real_market_data(symbol, "1y")
                market_analyzer.calculate_real_technical_indicators(symbol)
            
            # آموزش مدل‌ها
            for symbol in trading_config.SYMBOLS:
                logger.info(f"🤖 آموزش مدل {symbol}...")
                model_trainer.train_model(symbol)
            
            # اجرای بک‌تست اولیه
            for symbol in trading_config.SYMBOLS:
                logger.info(f"📊 اجرای بک‌تست {symbol}...")
                result = backtest_engine.run_backtest(
                    symbol, "2023-01-01", "2023-12-31"
                )
                if result:
                    logger.info(f"📈 نتایج بک‌تست {symbol}: بازده {result.total_return:.2f}%")
            
            self.is_initialized = True
            logger.info("✅ سیستم تریدینگ واقعی راه‌اندازی شد")
            
        except Exception as e:
            logger.error(f"❌ خطا در راه‌اندازی سیستم: {e}")
    
    def get_real_time_signal(self, symbol: str) -> Dict:
        """دریافت سیگنال واقعی"""
        if not self.is_initialized:
            self.initialize_system()
        
        try:
            # دریافت داده‌های لحظه‌ای
            current_data = market_analyzer.calculate_real_technical_indicators(symbol)
            
            if not current_data:
                return {'error': 'داده‌های کافی موجود نیست'}
            
            # پیش‌بینی سیگنال
            signal = model_trainer.predict_real_time(symbol, current_data)
            
            # ذخیره سیگنال
            trading_db.save_ai_signal(signal)
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ خطا در دریافت سیگنال {symbol}: {e}")
            return {'error': str(e)}

# ایجاد نمونه سیستم
trading_system = RealTradingSystem()

if __name__ == "__main__":
    # تست سیستم
    system = RealTradingSystem()
    system.initialize_system()
    
    # دریافت سیگنال واقعی
    signal = system.get_real_time_signal('bitcoin')
    print(f"🎯 سیگنال واقعی: {signal}")
