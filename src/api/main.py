# 📁 main.py

import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

from src.data.data_manager import SmartDataManager
from src.data.processing_pipeline import DataProcessingPipeline
from src.core.spiking_transformer.transformer_block import SpikingTransformerBlock
from src.core.technical_analysis.signal_engine import IntelligentSignalEngine
from src.core.risk_management.position_sizing import DynamicPositionSizing
from src.core.multi_timeframe.timeframe_sync import MultiTimeframeAnalyzer
from src.ai_ml.regime_classifier import MarketRegimeClassifier
from src.ai_ml.pattern_predictor import PatternPredictor
from src.backtesting.walk_forward import WalkForwardAnalyzer
from src.visualization.dashboard_builder import TradingDashboard
from src.monitoring.health_check import SystemHealthChecker
from src.utils.performance_tracker import PerformanceTracker
from src.utils.memory_monitor import MemoryMonitor

class CryptoAnalysisEngine:
    """موتور اصلی تحلیل بازار کریپتو - یکپارچه‌سازی تمام کامپوننت‌ها"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = self._setup_logging()
        self.performance_tracker = PerformanceTracker()
        self.memory_monitor = MemoryMonitor()
        self.health_checker = SystemHealthChecker()
        
        # کامپوننت‌های اصلی
        self.data_manager = None
        self.processing_pipeline = None
        self.spiking_transformer = None
        self.signal_engine = None
        self.risk_manager = None
        self.multi_timeframe_analyzer = None
        self.regime_classifier = None
        self.pattern_predictor = None
        self.backtester = None
        self.dashboard = None
        
        self.is_running = False
        
    def _setup_logging(self):
        """تنظیمات لاگینگ"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/crypto_engine.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger(__name__)
    
    async def initialize(self):
        """مقداردهی اولیه تمام کامپوننت‌ها"""
        self.logger.info("🔄 Initializing Crypto Analysis Engine...")
        
        try:
            # ۱. مدیریت داده
            with self.performance_tracker.track("data_manager_init"):
                self.data_manager = SmartDataManager(
                    api_key=self.config.get('api_key')
                )
            
            # ۲. پایپ‌لاین پردازش
            with self.performance_tracker.track("pipeline_init"):
                self.processing_pipeline = DataProcessingPipeline()
            
            # ۳. Spiking Transformer
            with self.performance_tracker.track("transformer_init"):
                self.spiking_transformer = SpikingTransformerBlock(
                    d_model=self.config['transformer']['d_model'],
                    n_heads=self.config['transformer']['n_heads'],
                    seq_len=self.config['transformer']['seq_len']
                )
            
            # ۴. موتور سیگنال
            with self.performance_tracker.track("signal_engine_init"):
                self.signal_engine = IntelligentSignalEngine()
            
            # ۵. مدیریت ریسک
            with self.performance_tracker.track("risk_manager_init"):
                self.risk_manager = DynamicPositionSizing(
                    total_capital=self.config['risk']['total_capital'],
                    max_risk_per_trade=self.config['risk']['max_risk_per_trade']
                )
            
            # ۶. تحلیل چندزمانه
            with self.performance_tracker.track("multi_timeframe_init"):
                self.multi_timeframe_analyzer = MultiTimeframeAnalyzer()
            
            # ۷. مدل‌های هوش مصنوعی
            with self.performance_tracker.track("ai_models_init"):
                self.regime_classifier = MarketRegimeClassifier()
                self.pattern_predictor = PatternPredictor()
                
                # لود مدل‌های از پیش آموزش دیده
                if not self.regime_classifier.load_model():
                    self.logger.warning("⚠️ Could not load pre-trained regime classifier")
                if not self.pattern_predictor.load_model():
                    self.logger.warning("⚠️ Could not load pre-trained pattern predictor")
            
            # ۸. سیستم بک‌تست
            with self.performance_tracker.track("backtester_init"):
                self.backtester = WalkForwardAnalyzer(
                    initial_capital=self.config['backtesting']['initial_capital']
                )
            
            # ۹. داشبورد
            with self.performance_tracker.track("dashboard_init"):
                self.dashboard = TradingDashboard()
            
            self.logger.info("✅ All components initialized successfully")
            self.health_checker.update_component_status("all_components", "healthy")
            
        except Exception as e:
            self.logger.error(f"❌ Initialization failed: {e}")
            self.health_checker.update_component_status("all_components", "failed")
            raise
    
    async def run_analysis_pipeline(self, symbols: List[str] = None) -> Dict:
        """اجرای کامل پایپ‌لاین تحلیل برای نمادها"""
        if symbols is None:
            symbols = self.config['default_symbols']
        
        self.logger.info(f"🔍 Starting analysis pipeline for {len(symbols)} symbols")
        
        analysis_results = {}
        
        for symbol in symbols:
            try:
                symbol_result = await self._analyze_symbol(symbol)
                analysis_results[symbol] = symbol_result
                
                self.logger.info(f"✅ Analysis completed for {symbol}")
                
            except Exception as e:
                self.logger.error(f"❌ Analysis failed for {symbol}: {e}")
                analysis_results[symbol] = {'error': str(e)}
        
        # ایجاد گزارش کلی
        summary = self._generate_analysis_summary(analysis_results)
        
        self.logger.info(f"📊 Analysis pipeline completed. Processed {len(analysis_results)} symbols")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'symbols_analyzed': list(analysis_results.keys()),
            'results': analysis_results,
            'summary': summary,
            'performance_metrics': self.performance_tracker.get_summary(),
            'memory_usage': self.memory_monitor.get_usage_stats()
        }
    
    async def _analyze_symbol(self, symbol: str) -> Dict:
        """آنالیز کامل یک نماد"""
        symbol_result = {}
        
        # ۱. دریافت و پردازش داده
        with self.performance_tracker.track(f"data_fetch_{symbol}"):
            raw_data = self.data_manager.get_coins_data(symbols=[symbol], limit=100)
            if not raw_data:
                raise Exception(f"No data available for {symbol}")
            
            processed_data = self.processing_pipeline.process_raw_data(raw_data)
        
        # ۲. تحلیل چندزمانه
        with self.performance_tracker.track(f"multi_tf_analysis_{symbol}"):
            multi_tf_data = self._prepare_multi_timeframe_data(symbol)
            mt_analysis = self.multi_timeframe_analyzer.analyze_symbol(symbol, multi_tf_data)
            symbol_result['multi_timeframe'] = mt_analysis
        
        # ۳. پیش‌بینی با Spiking Transformer
        with self.performance_tracker.track(f"transformer_prediction_{symbol}"):
            transformer_input = self._prepare_transformer_input(processed_data)
            with self.memory_monitor.track("transformer_inference"):
                transformer_output = self.spiking_transformer(transformer_input)
                symbol_result['transformer_prediction'] = {
                    'output': transformer_output.detach().numpy().tolist(),
                    'spike_stats': self.spiking_transformer.get_spike_statistics()
                }
        
        # ۴. تشخیص رژیم بازار
        with self.performance_tracker.track(f"regime_classification_{symbol}"):
            regime_prediction = self.regime_classifier.predict_regime(processed_data)
            symbol_result['market_regime'] = regime_prediction
        
        # ۵. پیش‌بینی الگو
        with self.performance_tracker.track(f"pattern_prediction_{symbol}"):
            pattern_prediction = self.pattern_predictor.predict_pattern(processed_data)
            symbol_result['pattern_prediction'] = pattern_prediction
        
        # ۶. تولید سیگنال
        with self.performance_tracker.track(f"signal_generation_{symbol}"):
            market_data = {symbol: processed_data}
            technical_indicators = self._extract_technical_indicators(processed_data)
            
            signals = self.signal_engine.generate_signals(market_data, technical_indicators)
            symbol_result['signals'] = [s.__dict__ for s in signals if s.symbol == symbol]
        
        # ۷. مدیریت ریسک برای سیگنال‌ها
        with self.performance_tracker.track(f"risk_assessment_{symbol}"):
            risk_assessments = []
            for signal in signals:
                if signal.symbol == symbol:
                    position_size = self.risk_manager.calculate_position_size(
                        signal, market_data
                    )
                    risk_assessments.append(position_size.__dict__)
            
            symbol_result['risk_assessment'] = risk_assessments
        
        return symbol_result
    
    def _prepare_multi_timeframe_data(self, symbol: str) -> Dict:
        """آماده‌سازی داده‌های چندزمانه"""
        # در نسخه واقعی از داده‌های واقعی استفاده می‌شود
        # اینجا یک نمونه ساده برگردانده می‌شود
        return {
            '1h': pd.DataFrame({
                'open': [50000, 50100, 50200],
                'high': [50500, 50600, 50700],
                'low': [49500, 49600, 49700],
                'close': [50200, 50300, 50400],
                'volume': [1000, 1200, 1100]
            }),
            '4h': pd.DataFrame({
                'open': [50000, 50200],
                'high': [50800, 50900],
                'low': [49400, 49500],
                'close': [50700, 50400],
                'volume': [4500, 4300]
            })
        }
    
    def _prepare_transformer_input(self, processed_data):
        """آماده‌سازی ورودی برای Transformer"""
        import torch
        # تبدیل داده پردازش شده به تنسور
        features = []
        for coin in processed_data.get('result', []):
            feature_vector = []
            for key, value in coin.items():
                if isinstance(value, (int, float)):
                    feature_vector.append(value)
            features.append(feature_vector)
        
        if not features:
            features = [[0] * 20]  # داده پیش‌فرض
        
        return torch.tensor([features], dtype=torch.float32)
    
    def _extract_technical_indicators(self, processed_data):
        """استخراج اندیکاتورهای تکنیکال از داده پردازش شده"""
        # اینجا می‌توان اندیکاتورهای واقعی را محاسبه کرد
        return {
            'rsi': pd.Series([45, 50, 55]),
            'macd': pd.Series([10, 12, 15]),
            'macd_signal': pd.Series([8, 10, 12]),
            'bollinger_upper': pd.Series([51000, 51200, 51400]),
            'bollinger_lower': pd.Series([49000, 49200, 49400])
        }
    
    def _generate_analysis_summary(self, analysis_results: Dict) -> Dict:
        """تولید گزارش خلاصه تحلیل"""
        total_symbols = len(analysis_results)
        successful_analysis = len([r for r in analysis_results.values() if 'error' not in r])
        
        # جمع‌آوری سیگنال‌ها
        all_signals = []
        for symbol, result in analysis_results.items():
            if 'signals' in result:
                all_signals.extend(result['signals'])
        
        buy_signals = [s for s in all_signals if s.get('signal_type') in ['BUY', 'STRONG_BUY']]
        sell_signals = [s for s in all_signals if s.get('signal_type') in ['SELL', 'STRONG_SELL']]
        
        return {
            'total_symbols_analyzed': total_symbols,
            'successful_analysis': successful_analysis,
            'success_rate': (successful_analysis / total_symbols * 100) if total_symbols > 0 else 0,
            'total_signals': len(all_signals),
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'strong_buy_signals': len([s for s in buy_signals if s.get('signal_type') == 'STRONG_BUY']),
            'strong_sell_signals': len([s for s in sell_signals if s.get('signal_type') == 'STRONG_SELL']),
            'timestamp': datetime.now().isoformat()
        }
    
    async def run_backtest(self, strategy_config: Dict) -> Dict:
        """اجرای بک‌تست برای استراتژی"""
        self.logger.info("🧪 Running strategy backtest...")
        
        # تولید داده برای بک‌تست
        backtest_data = self._generate_backtest_data()
        
        # اجرای Walk-Forward Analysis
        result = self.backtester.run_walk_forward_analysis(
            strategy=self._load_strategy(strategy_config),
            data=backtest_data,
            window_size=strategy_config.get('window_size', 100),
            step_size=strategy_config.get('step_size', 20)
        )
        
        # ایجاد داشبورد
        dashboard = self.dashboard.create_performance_dashboard(result['window_results'][0]['result'])
        
        return {
            'backtest_results': result,
            'dashboard': dashboard.to_json() if hasattr(dashboard, 'to_json') else str(dashboard)
        }
    
    def _generate_backtest_data(self):
        """تولید داده برای بک‌تست"""
        # در نسخه واقعی از داده‌های تاریخی واقعی استفاده می‌شود
        dates = pd.date_range('2023-01-01', periods=1000, freq='1h')
        prices = 50000 + np.cumsum(np.random.randn(1000) * 100)
        
        return pd.DataFrame({
            'open': prices + np.random.randn(1000) * 10,
            'high': prices + np.abs(np.random.randn(1000) * 20),
            'low': prices - np.abs(np.random.randn(1000) * 20),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 1000)
        }, index=dates)
    
    def _load_strategy(self, config: Dict):
        """لود استراتژی بر اساس پیکربندی"""
        # اینجا می‌توان استراتژی‌های مختلف را لود کرد
        from src.core.technical_analysis.signal_engine import IntelligentSignalEngine
        return IntelligentSignalEngine()
    
    async def start(self):
        """شروع موتور تحلیل"""
        self.logger.info("🚀 Starting Crypto Analysis Engine...")
        self.is_running = True
        
        # تنظیم handler برای shutdown
        signal.signal(signal.SIGINT, self._shutdown_signal_handler)
        signal.signal(signal.SIGTERM, self._shutdown_signal_handler)
        
        try:
            await self.initialize()
            
            # حلقه اصلی
            while self.is_running:
                try:
                    # اجرای تحلیل دوره‌ای
                    analysis_results = await self.run_analysis_pipeline()
                    
                    # به‌روزرسانی سلامت سیستم
                    self.health_checker.record_successful_cycle()
                    
                    # خواب قبل از سیکل بعدی
                    await asyncio.sleep(self.config.get('analysis_interval', 300))  # 5 دقیقه
                    
                except Exception as e:
                    self.logger.error(f"❌ Error in main loop: {e}")
                    self.health_checker.record_failed_cycle()
                    await asyncio.sleep(60)  # خواب کوتاه قبل از تلاش مجدد
                    
        except Exception as e:
            self.logger.error(f"❌ Fatal error: {e}")
        finally:
            await self.shutdown()
    
    def _shutdown_signal_handler(self, signum, frame):
        """Handler برای shutdown"""
        self.logger.info(f"🛑 Received shutdown signal {signum}")
        self.is_running = False
    
    async def shutdown(self):
        """خاموش کردن تمیز موتور"""
        self.logger.info("🛑 Shutting down Crypto Analysis Engine...")
        self.is_running = False
        
        # ذخیره لاگ‌های نهایی
        self.performance_tracker.save_summary()
        self.memory_monitor.save_usage_stats()
        
        self.logger.info("✅ Crypto Analysis Engine shutdown complete")

# تابع اصلی برای اجرا
async def main():
    """تابع اصلی اجرای برنامه"""
    config = {
        'api_key': 'your_api_key_here',
        'default_symbols': ['BTC/USDT', 'ETH/USDT', 'ADA/USDT'],
        'transformer': {
            'd_model': 64,
            'n_heads': 4,
            'seq_len': 10
        },
        'risk': {
            'total_capital': 10000,
            'max_risk_per_trade': 0.02
        },
        'backtesting': {
            'initial_capital': 10000
        },
        'analysis_interval': 300  # 5 minutes
    }
    
    engine = CryptoAnalysisEngine(config)
    
    try:
        await engine.start()
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # ایجاد پوشه لاگ‌ها
    os.makedirs('logs', exist_ok=True)
    
    # اجرای برنامه
    asyncio.run(main())
