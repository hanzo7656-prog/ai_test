# 📁 src/core/engine.py

import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import torch

from ..data.data_manager import SmartDataManager
from ..data.processing_pipeline import DataProcessingPipeline
from .spiking_transformer.transformer_block import SpikingTransformerBlock
from .technical_analysis.signal_engine import IntelligentSignalEngine, TradingSignal
from .risk_management.position_sizing import DynamicPositionSizing, PositionSizingResult
from .multi_timeframe.timeframe_sync import MultiTimeframeAnalyzer, TimeFrame
from ..ai_ml.regime_classifier import MarketRegimeClassifier
from ..ai_ml.pattern_predictor import PatternPredictor
from ..backtesting.walk_forward import WalkForwardAnalyzer
from ..visualization.dashboard_builder import TradingDashboard
from ..monitoring.health_check import SystemHealthChecker
from ..utils.performance_tracker import PerformanceTracker
from ..utils.memory_monitor import MemoryMonitor

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
            symbol_result['raw_data_stats'] = {
                'data_points': len(raw_data.get('result', [])),
                'processing_time': processed_data.get('processing_stats', {})
            }
        
        # ۲. تحلیل چندزمانه
        with self.performance_tracker.track(f"multi_tf_analysis_{symbol}"):
            multi_tf_data = self._prepare_multi_timeframe_data(symbol)
            mt_analysis = self.multi_timeframe_analyzer.analyze_symbol(symbol, multi_tf_data)
            symbol_result['multi_timeframe'] = mt_analysis
        
        # ۳. پیش‌بینی با Spiking Transformer
        with self.performance_tracker.track(f"transformer_prediction_{symbol}"):
            try:
                transformer_input = self._prepare_transformer_input(processed_data)
                with self.memory_monitor.track("transformer_inference"):
                    with torch.no_grad():
                        transformer_output = self.spiking_transformer(transformer_input)
                    
                    symbol_result['transformer_prediction'] = {
                        'output': transformer_output.detach().numpy().tolist(),
                        'spike_stats': self.spiking_transformer.get_spike_statistics(),
                        'prediction_direction': 'BULLISH' if transformer_output[0, -1, 0] > 0.5 else 'BEARISH',
                        'confidence': float(transformer_output[0, -1, 0])
                    }
            except Exception as e:
                self.logger.warning(f"Transformer prediction failed for {symbol}: {e}")
                symbol_result['transformer_prediction'] = {'error': str(e)}
        
        # ۴. تشخیص رژیم بازار
        with self.performance_tracker.track(f"regime_classification_{symbol}"):
            try:
                regime_prediction = self.regime_classifier.predict_regime(processed_data)
                symbol_result['market_regime'] = regime_prediction
            except Exception as e:
                self.logger.warning(f"Regime classification failed for {symbol}: {e}")
                symbol_result['market_regime'] = {'error': str(e)}
        
        # ۵. پیش‌بینی الگو
        with self.performance_tracker.track(f"pattern_prediction_{symbol}"):
            try:
                pattern_prediction = self.pattern_predictor.predict_pattern(processed_data)
                symbol_result['pattern_prediction'] = pattern_prediction
            except Exception as e:
                self.logger.warning(f"Pattern prediction failed for {symbol}: {e}")
                symbol_result['pattern_prediction'] = {'error': str(e)}
        
        # ۶. تولید سیگنال
        with self.performance_tracker.track(f"signal_generation_{symbol}"):
            try:
                market_data = {symbol: processed_data}
                technical_indicators = self._extract_technical_indicators(processed_data)
                
                signals = self.signal_engine.generate_signals(market_data, {symbol: technical_indicators})
                symbol_signals = [s for s in signals if s.symbol == symbol]
                
                symbol_result['signals'] = [self._signal_to_dict(s) for s in symbol_signals]
            except Exception as e:
                self.logger.warning(f"Signal generation failed for {symbol}: {e}")
                symbol_result['signals'] = []
        
        # ۷. مدیریت ریسک برای سیگنال‌ها
        with self.performance_tracker.track(f"risk_assessment_{symbol}"):
            try:
                risk_assessments = []
                market_data = {symbol: self._prepare_multi_timeframe_data(symbol)['1h']}
                
                for signal in symbol_signals:
                    position_size = self.risk_manager.calculate_position_size(
                        signal, market_data
                    )
                    risk_assessments.append(self._position_to_dict(position_size))
                
                symbol_result['risk_assessment'] = risk_assessments
            except Exception as e:
                self.logger.warning(f"Risk assessment failed for {symbol}: {e}")
                symbol_result['risk_assessment'] = []
        
        # ۸. جمع‌بندی نهایی
        symbol_result['analysis_summary'] = self._generate_symbol_summary(symbol_result)
        symbol_result['timestamp'] = datetime.now().isoformat()
        
        return symbol_result
    
    def _prepare_multi_timeframe_data(self, symbol: str) -> Dict:
        """آماده‌سازی داده‌های چندزمانه"""
        # در نسخه واقعی از داده‌های واقعی استفاده می‌شود
        # اینجا یک نمونه ساده برای تست برگردانده می‌شود
        
        base_price = 50000
        volatility = 0.02
        
        # داده 1h
        hours_1h = 100
        prices_1h = base_price + np.cumsum(np.random.randn(hours_1h) * base_price * volatility)
        
        data_1h = pd.DataFrame({
            'open': prices_1h + np.random.randn(hours_1h) * 10,
            'high': prices_1h + np.abs(np.random.randn(hours_1h) * 20),
            'low': prices_1h - np.abs(np.random.randn(hours_1h) * 20),
            'close': prices_1h,
            'volume': np.random.randint(1000, 10000, hours_1h)
        }, index=pd.date_range(end=datetime.now(), periods=hours_1h, freq='1h'))
        
        # داده 4h (تعداد کمتر)
        hours_4h = 25
        prices_4h = base_price + np.cumsum(np.random.randn(hours_4h) * base_price * volatility * 2)
        
        data_4h = pd.DataFrame({
            'open': prices_4h + np.random.randn(hours_4h) * 20,
            'high': prices_4h + np.abs(np.random.randn(hours_4h) * 40),
            'low': prices_4h - np.abs(np.random.randn(hours_4h) * 40),
            'close': prices_4h,
            'volume': np.random.randint(5000, 20000, hours_4h)
        }, index=pd.date_range(end=datetime.now(), periods=hours_4h, freq='4h'))
        
        return {
            '1h': data_1h,
            '4h': data_4h
        }
    
    def _prepare_transformer_input(self, processed_data):
        """آماده‌سازی ورودی برای Transformer"""
        # تبدیل داده پردازش شده به تنسور
        features = []
        
        for coin in processed_data.get('result', []):
            feature_vector = []
            for key, value in coin.items():
                if isinstance(value, (int, float)) and not key.endswith('_id'):
                    feature_vector.append(value)
            # اگر ویژگی‌ها کم هستند، با صفر پر کنیم
            while len(feature_vector) < 20:
                feature_vector.append(0.0)
            features.append(feature_vector[:20])  # حداکثر 20 ویژگی
        
        if not features:
            # داده پیش‌فرض اگر هیچ داده‌ای نبود
            features = [[0.0] * 20 for _ in range(10)]
        
        # تبدیل به تنسور: (batch_size, seq_len, features)
        tensor_data = torch.tensor([features], dtype=torch.float32)
        return tensor_data
    
    def _extract_technical_indicators(self, processed_data):
        """استخراج اندیکاتورهای تکنیکال از داده پردازش شده"""
        if not processed_data.get('result'):
            return {}
        
        # استفاده از اولین کوین در نتایج
        coin_data = processed_data['result'][0]
        
        # شبیه‌سازی اندیکاتورها - در نسخه واقعی از محاسبات واقعی استفاده می‌شود
        return {
            'rsi': pd.Series([coin_data.get('price', 50000) / 1000 + 30]),  # RSI شبیه‌سازی شده
            'macd': pd.Series([coin_data.get('price', 50000) / 10000]),
            'macd_signal': pd.Series([coin_data.get('price', 50000) / 12000]),
            'bollinger_upper': pd.Series([coin_data.get('price', 50000) * 1.02]),
            'bollinger_lower': pd.Series([coin_data.get('price', 50000) * 0.98]),
            'volume': pd.Series([coin_data.get('volume', 1000)]),
            'price_change_1h': pd.Series([coin_data.get('price', 50000) / 50000 - 1])
        }
    
    def _signal_to_dict(self, signal: TradingSignal) -> Dict:
        """تبدیل سیگنال به دیکشنری"""
        return {
            'symbol': signal.symbol,
            'signal_type': signal.signal_type.value,
            'confidence': signal.confidence,
            'price': signal.price,
            'timestamp': signal.timestamp.isoformat(),
            'reasons': signal.reasons,
            'targets': signal.targets,
            'stop_loss': signal.stop_loss,
            'time_horizon': signal.time_horizon,
            'risk_reward_ratio': signal.risk_reward_ratio
        }
    
    def _position_to_dict(self, position: PositionSizingResult) -> Dict:
        """تبدیل موقعیت به دیکشنری"""
        return {
            'symbol': position.symbol,
            'position_size': position.position_size,
            'risk_amount': position.risk_amount,
            'stop_loss': position.stop_loss,
            'take_profit': position.take_profit,
            'leverage': position.leverage,
            'max_position_value': position.max_position_value
        }
    
    def _generate_symbol_summary(self, symbol_result: Dict) -> Dict:
        """تولید خلاصه تحلیل برای یک نماد"""
        summary = {
            'overall_score': 0.0,
            'recommendation': 'HOLD',
            'confidence': 0.0,
            'key_factors': []
        }
        
        # محاسبه امتیاز کلی
        scores = []
        
        # امتیاز از Transformer
        if 'transformer_prediction' in symbol_result and 'error' not in symbol_result['transformer_prediction']:
            transformer_conf = symbol_result['transformer_prediction'].get('confidence', 0)
            scores.append(transformer_conf)
            summary['key_factors'].append('AI Prediction')
        
        # امتیاز از سیگنال‌ها
        if symbol_result.get('signals'):
            best_signal = max(symbol_result['signals'], key=lambda x: x['confidence'])
            scores.append(best_signal['confidence'])
            summary['recommendation'] = best_signal['signal_type']
            summary['key_factors'].append('Technical Signals')
        
        # امتیاز از رژیم بازار
        if 'market_regime' in symbol_result and 'error' not in symbol_result['market_regime']:
            regime_conf = symbol_result['market_regime'].get('confidence', 0)
            scores.append(regime_conf)
            summary['key_factors'].append('Market Regime')
        
        if scores:
            summary['overall_score'] = sum(scores) / len(scores)
            summary['confidence'] = summary['overall_score']
        
        return summary
    
    def _generate_analysis_summary(self, analysis_results: Dict) -> Dict:
        """تولید گزارش خلاصه تحلیل"""
        total_symbols = len(analysis_results)
        successful_analysis = len([r for r in analysis_results.values() if 'error' not in r.get('analysis_summary', {})])
        
        # جمع‌آوری سیگنال‌ها و رژیم‌ها
        all_signals = []
        market_regimes = {}
        
        for symbol, result in analysis_results.items():
            if 'signals' in result:
                all_signals.extend(result['signals'])
            
            if 'market_regime' in result and 'error' not in result['market_regime']:
                regime = result['market_regime']['regime']
                market_regimes[regime] = market_regimes.get(regime, 0) + 1
        
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
            'market_regimes': market_regimes,
            'timestamp': datetime.now().isoformat(),
            'average_confidence': np.mean([s.get('confidence', 0) for s in all_signals]) if all_signals else 0
        }
    
    async def run_backtest(self, strategy_config: Dict) -> Dict:
        """اجرای بک‌تست برای استراتژی"""
        self.logger.info("🧪 Running strategy backtest...")
        
        try:
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
            if result['window_results']:
                dashboard = self.dashboard.create_performance_dashboard(result['window_results'][0]['result'])
                dashboard_html = "Dashboard generated successfully"
            else:
                dashboard_html = "No results to display"
            
            return {
                'backtest_id': f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'strategy_config': strategy_config,
                'results': result,
                'dashboard': dashboard_html,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            raise
    
    def _generate_backtest_data(self):
        """تولید داده برای بک‌تست"""
        dates = pd.date_range('2023-01-01', periods=1000, freq='1h')
        base_price = 50000
        returns = np.random.randn(1000) * 0.001  # بازدهی روزانه ~0.1%
        prices = base_price * np.exp(np.cumsum(returns))
        
        return pd.DataFrame({
            'open': prices + np.random.randn(1000) * 10,
            'high': prices + np.abs(np.random.randn(1000) * 20),
            'low': prices - np.abs(np.random.randn(1000) * 20),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 1000)
        }, index=dates)
    
    def _load_strategy(self, config: Dict):
        """لود استراتژی بر اساس پیکربندی"""
        # برای سادگی، از همان IntelligentSignalEngine استفاده می‌کنیم
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
                    
                    self.logger.info(f"📈 Analysis cycle completed. Signals: {analysis_results['summary']['total_signals']}")
                    
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

    def get_system_status(self) -> Dict:
        """دریافت وضعیت سیستم"""
        return {
            'status': 'running' if self.is_running else 'stopped',
            'components_healthy': self.health_checker.get_system_status(),
            'performance_metrics': self.performance_tracker.get_summary(),
            'memory_usage': self.memory_monitor.get_usage_stats(),
            'timestamp': datetime.now().isoformat()
        }
