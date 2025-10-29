# technical_scanner.py - اسکنر مبتنی بر معماری اسپارس با داده‌های خام

import torch
import numpy as np
from typing import List, Dict, Any
import logging
from datetime import datetime
from trading_ai.sparse_technical_analyzer import SparseTechnicalNetwork, SparseConfig
from trading_ai.complete_coinstats_manager import coin_stats_manager
from trading_ai.lbank_websocket import get_websocket_manager

logger = logging.getLogger(__name__)

class SparseTechnicalScanner:
    """اسکنر تکنیکال مبتنی بر معماری اسپارس با داده‌های خام"""
    
    def __init__(self, model_path: str = None):
        self.config = SparseConfig()
        
        if model_path:
            self.model = self.load_model(model_path)
        else:
            self.model = SparseTechnicalNetwork(self.config)
            
        self.ws_manager = get_websocket_manager()
        self.scan_results_cache = {}
        self.scan_config = {
            'min_confidence': 0.6,
            'max_symbols_per_scan': 50,
            'timeframe': '1d',
            'use_realtime_data': True
        }
        
        logger.info("🚀 Sparse Technical Scanner Initialized - Raw Data Mode")

    def load_model(self, model_path: str):
        """بارگذاری مدل از فایل"""
        try:
            checkpoint = torch.load(model_path)
            config = SparseConfig(**checkpoint['config'])
            model = SparseTechnicalNetwork(config)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            logger.info(f"✅ مدل اسکنر از {model_path} بارگذاری شد")
            return model
            
        except Exception as e:
            logger.error(f"❌ خطا در بارگذاری مدل: {e}")
            return SparseTechnicalNetwork(self.config)

    def scan_market(self, symbols: List[str], conditions: Dict = None) -> List[Dict]:
        """اسکن بازار با شرایط تکنیکال و داده‌های خام"""
        try:
            results = []
            conditions = conditions or {}
            
            logger.info(f"🔍 اسکن بازار برای {len(symbols)} نماد با داده‌های خام")

            for symbol in symbols[:self.scan_config['max_symbols_per_scan']]:
                try:
                    # دریافت داده‌های بازار خام
                    market_data = self.get_market_data(symbol)
                    
                    if not market_data:
                        continue

                    # تحلیل با مدل اسپارس
                    analysis = self.model.analyze_raw_market_data(market_data)
                    
                    # بررسی شرایط
                    if self.check_conditions(analysis, conditions):
                        results.append({
                            'symbol': symbol,
                            'analysis': analysis,
                            'timestamp': datetime.now().isoformat(),
                            'conditions_met': True,
                            'raw_data_quality': market_data.get('quality_metrics', {}),
                            'scan_confidence': analysis.get('model_confidence', 0.0)
                        })
                        
                        logger.info(f"✅ نماد {symbol} شرایط را دارد - اطمینان: {analysis.get('model_confidence', 0.0):.3f}")
                        
                except Exception as e:
                    logger.error(f"❌ خطا در اسکن نماد {symbol}: {e}")
                    continue

            # کش کردن نتایج
            scan_id = f"scan_{int(datetime.now().timestamp())}"
            self.scan_results_cache[scan_id] = {
                'results': results,
                'total_scanned': len(symbols),
                'symbols_found': len(results),
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"📊 اسکن بازار کامل شد: {len(results)} نماد از {len(symbols)} پیدا شد")
            return results

        except Exception as e:
            logger.error(f"❌ خطا در اسکن بازار: {e}")
            return []

    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """دریافت داده‌های بازار خام برای تحلیل"""
        try:
            market_data = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'data_sources': [],
                'quality_metrics': {}
            }

            # 1. دریافت داده‌های تاریخی از CoinStats
            try:
                historical_data = coin_stats_manager.get_coin_charts(
                    symbol, 
                    self.scan_config['timeframe']
                )
                
                if historical_data and 'result' in historical_data:
                    prices = []
                    volumes = []
                    
                    for item in historical_data['result']:
                        if 'price' in item:
                            try:
                                prices.append(float(item['price']))
                                # حجم ممکن است در برخی اندپوینت‌ها موجود نباشد
                                if 'volume' in item:
                                    volumes.append(float(item.get('volume', 0)))
                            except (ValueError, TypeError):
                                continue
                    
                    market_data['prices'] = prices
                    market_data['volumes'] = volumes if volumes else []
                    market_data['data_sources'].append('coinstats_historical')
                    
            except Exception as e:
                logger.debug(f"⚠️ خطا در دریافت داده‌های تاریخی {symbol}: {e}")

            # 2. دریافت داده‌های لحظه‌ای از WebSocket
            if self.scan_config['use_realtime_data']:
                try:
                    realtime_data = self.ws_manager.get_realtime_data(symbol)
                    if realtime_data:
                        market_data['realtime'] = realtime_data
                        market_data['data_sources'].append('websocket_realtime')
                except Exception as e:
                    logger.debug(f"⚠️ خطا در دریافت داده‌های لحظه‌ای {symbol}: {e}")

            # 3. دریافت داده‌های جاری از CoinStats
            try:
                current_data = coin_stats_manager.get_coin_details(symbol, "USD")
                if current_data and 'result' in current_data:
                    market_data['current'] = current_data['result']
                    market_data['data_sources'].append('coinstats_current')
            except Exception as e:
                logger.debug(f"⚠️ خطا در دریافت داده‌های جاری {symbol}: {e}")

            # محاسبه کیفیت داده‌های خام
            market_data['quality_metrics'] = self._calculate_data_quality(market_data)
            
            # بررسی حداقل داده مورد نیاز
            if len(market_data.get('prices', [])) < 10:
                logger.warning(f"⚠️ داده‌های ناکافی برای نماد {symbol}")
                return None

            return market_data

        except Exception as e:
            logger.error(f"❌ خطا در دریافت داده‌های بازار {symbol}: {e}")
            return None

    def _calculate_data_quality(self, market_data: Dict) -> Dict[str, float]:
        """محاسبه کیفیت داده‌های خام"""
        quality_metrics = {
            'completeness': 0.0,
            'freshness': 0.0,
            'consistency': 0.8,  # مقدار پیش‌فرض
            'overall_score': 0.0
        }

        try:
            # کامل بودن داده‌ها
            data_sources = market_data.get('data_sources', [])
            quality_metrics['completeness'] = min(len(data_sources) / 3, 1.0)

            # تازگی داده‌ها
            if 'realtime' in market_data or 'websocket_realtime' in data_sources:
                quality_metrics['freshness'] = 1.0
            elif 'coinstats_current' in data_sources:
                quality_metrics['freshness'] = 0.7
            else:
                quality_metrics['freshness'] = 0.3

            # سازگاری داده‌های قیمتی
            prices = market_data.get('prices', [])
            if len(prices) > 1:
                # بررسی اینکه قیمت‌ها منطقی هستند
                price_changes = np.diff(prices)
                extreme_changes = np.sum(np.abs(price_changes) > np.mean(prices) * 0.1)  # تغییرات بیش از 10%
                quality_metrics['consistency'] = max(0.5, 1.0 - (extreme_changes / len(price_changes)))

            # نمره کلی کیفیت
            quality_metrics['overall_score'] = round(
                (quality_metrics['completeness'] * 0.4 +
                 quality_metrics['freshness'] * 0.3 +
                 quality_metrics['consistency'] * 0.3), 3
            )

            quality_metrics['quality_level'] = (
                'high' if quality_metrics['overall_score'] > 0.8 else
                'medium' if quality_metrics['overall_score'] > 0.5 else 'low'
            )

        except Exception as e:
            logger.error(f"❌ خطا در محاسبه کیفیت داده: {e}")

        return quality_metrics

    def check_conditions(self, analysis: Dict, conditions: Dict) -> bool:
        """بررسی شرایط اسکن با داده‌های خام"""
        try:
            if not conditions:
                return analysis.get('model_confidence', 0) >= self.scan_config['min_confidence']

            conditions_met = 0
            total_conditions = len(conditions)

            # بررسی شرایط روند
            if 'min_trend_confidence' in conditions:
                trend_conf = analysis.get('trend_analysis', {}).get('confidence', 0)
                if trend_conf >= conditions['min_trend_confidence']:
                    conditions_met += 1

            # بررسی شرایط الگو
            if 'required_pattern' in conditions:
                pattern = analysis.get('pattern_analysis', {}).get('detected_pattern', '')
                if pattern == conditions['required_pattern']:
                    conditions_met += 1

            # بررسی شرایط نوسان
            if 'max_volatility' in conditions:
                volatility = analysis.get('market_metrics', {}).get('volatility', 0)
                if volatility <= conditions['max_volatility']:
                    conditions_met += 1

            # بررسی اطمینان کلی
            model_confidence = analysis.get('model_confidence', 0)
            if model_confidence >= self.scan_config['min_confidence']:
                conditions_met += 1

            # اگر همه شرایط الزامی هستند
            if conditions.get('require_all', False):
                return conditions_met == total_conditions
            else:
                return conditions_met >= max(1, total_conditions // 2)

        except Exception as e:
            logger.error(f"❌ خطا در بررسی شرایط: {e}")
            return False

    def get_technical_recommendations(self, analysis: Dict) -> List[str]:
        """تولید توصیه های تحلیلی از داده‌های خام"""
        recommendations = []
        
        try:
            trend_analysis = analysis.get('trend_analysis', {})
            pattern_analysis = analysis.get('pattern_analysis', {})
            market_metrics = analysis.get('market_metrics', {})
            
            trend = trend_analysis.get('direction', 'خنثی')
            trend_confidence = trend_analysis.get('confidence', 0)
            pattern = pattern_analysis.get('detected_pattern', 'هیچ')
            volatility = market_metrics.get('volatility', 0)
            overall_confidence = analysis.get('model_confidence', 0)

            # توصیه‌های مبتنی بر روند
            if trend == 'صعودی' and trend_confidence > 0.7:
                recommendations.append("📈 روند صعودی قوی - فرصت خرید")
                recommendations.append("🎯 هدف: سطوح مقاومتی بالاتر")
            elif trend == 'نزولی' and trend_confidence > 0.7:
                recommendations.append("📉 روند نزولی قوی - احتیاط در خرید")
                recommendations.append("🛡️ حد ضرر محافظه‌کارانه تنظیم شود")
            else:
                recommendations.append("⚖️ بازار در حالت خنثی - انتظار برای شکست")

            # توصیه‌های مبتنی بر الگو
            if pattern != 'هیچ':
                pattern_confidence = pattern_analysis.get('confidence', 0)
                if pattern_confidence > 0.6:
                    recommendations.append(f"🎯 الگوی {pattern} شناسایی شد")
                    
                    if pattern in ['سر و شانه', 'دو قله']:
                        recommendations.append("⚠️ احتمال بازگشت روند وجود دارد")
                    elif pattern in ['دو دره', 'مثلث']:
                        recommendations.append("💡 احتمال ادامه روند وجود دارد")

            # توصیه‌های مبتنی بر نوسان
            if volatility > 0.1:
                recommendations.append("🌊 نوسان بالا - مدیریت ریسک ضروری")
                recommendations.append("📊 استفاده از حجم معاملات کمتر")
            else:
                recommendations.append("🌊 نوسان متوسط - شرایط معاملاتی نرمال")

            # توصیه کلی بر اساس اطمینان مدل
            if overall_confidence > 0.8:
                recommendations.append("✅ تحلیل با اطمینان بالا - قابل اتکا")
            elif overall_confidence > 0.6:
                recommendations.append("⚠️ تحلیل با اطمینان متوسط - احتیاط لازم")
            else:
                recommendations.append("❌ اطمینان تحلیل پایین - بررسی مجدد")

            # افزودن یادداشت درباره داده‌های خام
            recommendations.append("🔍 تحلیل مبتنی بر داده‌های خام بازار")

        except Exception as e:
            logger.error(f"❌ خطا در تولید توصیه‌ها: {e}")
            recommendations.append("⚠️ خطا در تولید توصیه‌های تحلیلی")

        return recommendations

    def get_scanner_status(self) -> Dict[str, Any]:
        """دریافت وضعیت اسکنر"""
        return {
            'model_loaded': self.model is not None,
            'config': self.config.__dict__,
            'scan_config': self.scan_config,
            'last_scan_time': list(self.scan_results_cache.keys())[-1] if self.scan_results_cache else None,
            'total_cached_scans': len(self.scan_results_cache),
            'websocket_connected': self.ws_manager.is_connected() if self.ws_manager else False,
            'raw_data_mode': True,
            'data_sources': ['CoinStats', 'WebSocket']
        }

    def clear_cache(self):
        """پاکسازی کش نتایج اسکن"""
        self.scan_results_cache.clear()
        logger.info("✅ کش نتایج اسکن پاکسازی شد")

    def get_scan_history(self, limit: int = 10) -> List[Dict]:
        """دریافت تاریخچه اسکن‌ها"""
        scans = list(self.scan_results_cache.items())
        scans.sort(key=lambda x: x[0], reverse=True)  # مرتب‌سازی بر اساس زمان
        
        history = []
        for scan_id, scan_data in scans[:limit]:
            history.append({
                'scan_id': scan_id,
                'timestamp': scan_data['timestamp'],
                'total_scanned': scan_data['total_scanned'],
                'symbols_found': scan_data['symbols_found']
            })
            
        return history

# ایجاد نمونه گلوبال
technical_scanner = SparseTechnicalScanner()
