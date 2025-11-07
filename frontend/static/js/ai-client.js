// کلاینت هوش مصنوعی VortexAI - نسخه اصلاح شده و سازگار با بک‌اند
class AIClient {
    constructor() {
        this.isInitialized = false;
        this.models = {
            technical: null,
            sentiment: null,
            predictive: null
        };
        this.analysisHistory = [];
        this.apiBase = '/api/ai';
        this.cache = new Map();
        this.cacheTTL = 5 * 60 * 1000; // 5 دقیقه
        
        // آمار استفاده
        this.usageStats = {
            totalRequests: 0,
            successfulRequests: 0,
            failedRequests: 0,
            averageResponseTime: 0,
            lastRequestTime: null
        };

        console.log('✅ AI Client initialized');
    }

    async initialize() {
        try {
            console.log('🚀 Starting AI engine initialization...');
            
            // تست اتصال به AI backend از طریق روت سلامت
            const startTime = Date.now();
            const response = await fetch('/api/ai/status');
            
            if (!response.ok) {
                throw new Error(`AI status check failed: ${response.status}`);
            }

            const status = await response.json();
            const responseTime = Date.now() - startTime;
            
            this.updateUsageStats(true, responseTime);

            // ✅ اصلاح شده: تطابق با ساختار واقعی backend
            if (status.status === 'operational') {
                this.isInitialized = true;
                this.models = status.modules || {};
                
                console.log('✅ AI Client initialized successfully', {
                    models: this.models,
                    status: status.status
                });
                
                return true;
            } else {
                throw new Error(`AI backend status: ${status.status}`);
            }
        } catch (error) {
            console.error('❌ AI Client initialization failed:', error);
            this.updateUsageStats(false, 0);
            
            // Fallback به حالت شبیه‌سازی برای توسعه
            return this.initializeFallback();
        }
    }

    async initializeFallback() {
        // شبیه‌سازی برای زمانی که AI backend در دسترس نیست
        this.models = {
            neural_network: { 
                active: false, 
                neurons: 100, 
                sparsity: "80.0%", 
                trained: false 
            },
            technical_analysis: {
                rsi_analyzer: false,
                macd_analyzer: false,
                signal_generator: false
            },
            data_processing: false
        };
        this.isInitialized = true;
        console.log('🔶 AI Client running in fallback mode');
        return true;
    }

    async analyzeTechnical(symbol, data = null) {
        const startTime = Date.now();
        
        try {
            if (!this.isInitialized) {
                await this.initialize();
            }

            // بررسی کش
            const cacheKey = `technical_${symbol}`;
            const cached = this.getFromCache(cacheKey);
            if (cached) {
                console.log(`📦 Using cached technical analysis for ${symbol}`);
                return cached;
            }

            console.log(`🧠 Starting technical analysis for ${symbol}`);

            const response = await fetch(`${this.apiBase}/analyze/${symbol}?analysis_type=technical`, {
                method: 'GET',
                headers: {
                    'Accept': 'application/json',
                    'Cache-Control': 'no-cache'
                }
            });

            if (!response.ok) {
                throw new Error(`AI analysis failed: ${response.status} ${response.statusText}`);
            }

            const result = await response.json();
            const responseTime = Date.now() - startTime;
            
            this.updateUsageStats(true, responseTime);

            // ✅ اصلاح شده: مدیریت خطاهای تحلیل تکنیکال
            if (result.analysis && result.analysis.error) {
                console.warn(`⚠️ AI analysis error for ${symbol}: ${result.analysis.error}`);
                // استفاده از fallback اگر خطا وجود دارد
                return this.fallbackTechnicalAnalysis(data, symbol);
            }

            // ذخیره در تاریخچه
            this.analysisHistory.push({
                symbol: symbol,
                analysis: result,
                timestamp: new Date().toISOString(),
                type: 'technical',
                responseTime: responseTime
            });

            // ذخیره در کش
            this.setToCache(cacheKey, result);

            console.log(`✅ Technical analysis completed for ${symbol} in ${responseTime}ms`);
            return result;

        } catch (error) {
            const responseTime = Date.now() - startTime;
            this.updateUsageStats(false, responseTime);
            
            console.error(`❌ Technical analysis failed for ${symbol}:`, error);
            // Fallback به تحلیل ساده
            return this.fallbackTechnicalAnalysis(data, symbol);
        }
    }

    async analyzeSentiment(symbol, data = null) {
        const startTime = Date.now();
        
        try {
            console.log(`😊 Starting sentiment analysis for ${symbol}`);

            // بررسی کش
            const cacheKey = `sentiment_${symbol}`;
            const cached = this.getFromCache(cacheKey);
            if (cached) {
                console.log(`📦 Using cached sentiment analysis for ${symbol}`);
                return cached;
            }

            const response = await fetch(`${this.apiBase}/analyze/${symbol}?analysis_type=sentiment`, {
                method: 'GET',
                headers: {
                    'Accept': 'application/json',
                    'Cache-Control': 'no-cache'
                }
            });

            if (!response.ok) {
                throw new Error(`Sentiment analysis failed: ${response.status}`);
            }

            const result = await response.json();
            const responseTime = Date.now() - startTime;
            
            this.updateUsageStats(true, responseTime);

            this.analysisHistory.push({
                symbol: symbol,
                analysis: result,
                timestamp: new Date().toISOString(),
                type: 'sentiment',
                responseTime: responseTime
            });

            this.setToCache(cacheKey, result);

            console.log(`✅ Sentiment analysis completed for ${symbol} in ${responseTime}ms`);
            return result;

        } catch (error) {
            const responseTime = Date.now() - startTime;
            this.updateUsageStats(false, responseTime);
            
            console.error(`❌ Sentiment analysis failed for ${symbol}:`, error);
            return this.fallbackSentimentAnalysis(symbol);
        }
    }

    async getPrediction(symbol, period = '1d', data = null) {
        const startTime = Date.now();
        
        try {
            console.log(`🔮 Starting price prediction for ${symbol} (${period})`);

            // بررسی کش
            const cacheKey = `prediction_${symbol}_${period}`;
            const cached = this.getFromCache(cacheKey);
            if (cached) {
                console.log(`📦 Using cached prediction for ${symbol}`);
                return cached;
            }

            const response = await fetch(`${this.apiBase}/analyze/${symbol}?analysis_type=prediction`, {
                method: 'GET',
                headers: {
                    'Accept': 'application/json',
                    'Cache-Control': 'no-cache'
                }
            });

            if (!response.ok) {
                throw new Error(`Prediction failed: ${response.status}`);
            }

            const result = await response.json();
            const responseTime = Date.now() - startTime;
            
            this.updateUsageStats(true, responseTime);

            this.analysisHistory.push({
                symbol: symbol,
                analysis: result,
                timestamp: new Date().toISOString(),
                type: 'prediction',
                period: period,
                responseTime: responseTime
            });

            this.setToCache(cacheKey, result);

            console.log(`✅ Prediction completed for ${symbol} in ${responseTime}ms`);
            return result;

        } catch (error) {
            const responseTime = Date.now() - startTime;
            this.updateUsageStats(false, responseTime);
            
            console.error(`❌ Prediction failed for ${symbol}:`, error);
            return this.fallbackPrediction(symbol, period);
        }
    }

    // ===== متدهای Fallback پیشرفته =====
    fallbackTechnicalAnalysis(data, symbol) {
        console.log(`🔄 Using fallback technical analysis for ${symbol}`);
        
        // ✅ بهبود: استفاده از داده‌های واقعی اگر موجود باشد
        let price = 0;
        let change = 0;
        let volume = 0;
        let marketCap = 0;
        
        // استخراج داده از ساختارهای مختلف API
        if (data && data.data) {
            const responseData = data.data;
            
            if (responseData.market_data) {
                const market = responseData.market_data;
                price = market.price || market.current_price || 0;
                change = market.priceChange1d || market.price_change_24h || 0;
                volume = market.volume || market.total_volume || 0;
                marketCap = market.marketCap || market.market_cap || 0;
            } else if (responseData.display_data) {
                const display = responseData.display_data;
                price = display.price || display.current_price || 0;
                change = display.price_change_24h || display.priceChange1d || 0;
                volume = display.volume_24h || display.total_volume || 0;
                marketCap = display.market_cap || display.marketCap || 0;
            }
        } else if (data && (data.price !== undefined)) {
            price = data.price || 0;
            change = data.priceChange1d || data.price_change_24h || 0;
            volume = data.volume || 0;
            marketCap = data.marketCap || 0;
        }

        // اگر داده‌ای موجود نبود، از داده تست استفاده کن
        if (price === 0) {
            const hash = this.stringToHash(symbol);
            price = 1000 + (hash % 50000);
            change = (hash % 40) - 20;
            volume = 1000000 + (hash % 100000000);
            marketCap = 10000000 + (hash % 1000000000);
        }

        // محاسبه RSI ساده
        const rsi = this.calculateSimpleRSI(change);
        
        // تحلیل روند پیشرفته‌تر
        let signal = 'HOLD';
        let confidence = 0.5;

        if (rsi < 25 && change > 2) {
            signal = 'STRONG_BUY';
            confidence = 0.85;
        } else if (rsi < 35 && change > 0) {
            signal = 'BUY';
            confidence = 0.65;
        } else if (rsi > 75 && change < -2) {
            signal = 'STRONG_SELL';
            confidence = 0.85;
        } else if (rsi > 65 && change < 0) {
            signal = 'SELL';
            confidence = 0.65;
        } else if (Math.abs(change) < 1) {
            signal = 'HOLD';
            confidence = 0.7; // نوسان کم = اعتماد بیشتر
        }

        // افزایش confidence بر اساس حجم و مارکت کپ
        if (volume > 1000000000) {
            confidence = Math.min(0.95, confidence + 0.15);
        }
        if (marketCap > 10000000000) {
            confidence = Math.min(0.95, confidence + 0.1);
        }

        const analysis = {
            status: 'success',
            symbol: symbol,
            analysis_type: 'technical',
            analysis: {
                signal: signal,
                confidence: confidence,
                indicators: {
                    rsi: Math.round(rsi),
                    trend: change > 0 ? 'صعودی' : 'نزولی',
                    volume_impact: volume > 1000000000 ? 'بالا' : 'عادی',
                    price_change_24h: change,
                    market_cap_impact: marketCap > 10000000000 ? 'بالا' : 'عادی',
                    volatility: Math.min(100, Math.abs(change) * 3)
                },
                summary: this.generateSummary(signal, confidence, rsi, change),
                timestamp: new Date().toISOString()
            },
            source: 'fallback',
            fallback: true
        };

        console.log(`🔶 Fallback technical analysis for ${symbol}:`, analysis.analysis);
        return analysis;
    }

    fallbackSentimentAnalysis(symbol) {
        const sentiments = ['SLIGHTLY_BULLISH', 'NEUTRAL', 'SLIGHTLY_BEARISH'];
        const randomSentiment = sentiments[Math.floor(Math.random() * sentiments.length)];
        
        const sentiment = {
            status: 'success',
            symbol: symbol,
            analysis_type: 'sentiment',
            analysis: {
                sentiment: randomSentiment,
                confidence: 0.4 + Math.random() * 0.3,
                indicators: {
                    social_volume: 'medium',
                    news_sentiment: 'neutral',
                    market_mood: Math.random() > 0.5 ? 'bullish' : 'bearish',
                    price_momentum: 'stable'
                },
                summary: 'تحلیل احساسات در دسترس نیست - استفاده از تحلیل پایه',
                timestamp: new Date().toISOString()
            },
            source: 'fallback',
            fallback: true
        };

        return sentiment;
    }

    fallbackPrediction(symbol, period) {
        const basePrice = 1000 + (this.stringToHash(symbol) % 50000);
        const volatility = 0.02 + (Math.random() * 0.03);
        const predictedPrice = Math.round(basePrice * (1 + (Math.random() - 0.5) * volatility));
        const currentPrice = basePrice;
        const direction = predictedPrice > currentPrice ? 'UP' : 'DOWN';
        const changePercent = ((predictedPrice - currentPrice) / currentPrice * 100).toFixed(2);
        
        return {
            status: 'success',
            symbol: symbol,
            analysis_type: 'prediction',
            analysis: {
                predicted_price: predictedPrice,
                price_change_percent: parseFloat(changePercent),
                current_price: currentPrice,
                direction: direction,
                confidence: 0.3 + Math.random() * 0.4,
                time_frame: period,
                neural_network_used: false,
                note: 'مدل آموزش ندیده - استفاده از پیش‌بینی پایه',
                timestamp: new Date().toISOString()
            },
            source: 'fallback',
            fallback: true,
            disclaimer: 'پیش‌بینی بر اساس داده‌های محدود - برای تحلیل دقیق‌تر از سرور AI استفاده کنید'
        };
    }

    // ===== ابزارهای تحلیل پیشرفته =====
    calculateSimpleRSI(change) {
        // شبیه‌سازی RSI بر اساس تغییرات قیمت
        return Math.min(100, Math.max(0, 50 + (change * 1.5)));
    }

    generateSummary(signal, confidence, rsi, change) {
        const parts = [];
        
        if (signal.includes('STRONG_BUY')) {
            parts.push('سیگنال خرید قوی');
        } else if (signal.includes('BUY')) {
            parts.push('سیگنال خرید');
        } else if (signal.includes('STRONG_SELL')) {
            parts.push('سیگنال فروش قوی');
        } else if (signal.includes('SELL')) {
            parts.push('سیگنال فروش');
        } else {
            parts.push('سیگنال نگهداری');
        }

        if (confidence > 0.8) {
            parts.push('اعتماد بسیار بالا');
        } else if (confidence > 0.6) {
            parts.push('اعتماد بالا');
        } else if (confidence > 0.4) {
            parts.push('اعتماد متوسط');
        } else {
            parts.push('اعتماد پایین');
        }

        if (rsi < 25) {
            parts.push('اشباع فروش شدید');
        } else if (rsi < 35) {
            parts.push('اشباع فروش');
        } else if (rsi > 75) {
            parts.push('اشباع خرید شدید');
        } else if (rsi > 65) {
            parts.push('اشباع خرید');
        }

        if (Math.abs(change) > 15) {
            parts.push('نوسان بسیار شدید');
        } else if (Math.abs(change) > 8) {
            parts.push('نوسان شدید');
        } else if (Math.abs(change) > 3) {
            parts.push('نوسان متوسط');
        }

        return parts.join(' • ');
    }

    // ===== سیستم کش پیشرفته =====
    getFromCache(key) {
        const item = this.cache.get(key);
        if (!item) return null;

        if (Date.now() > item.expiry) {
            this.cache.delete(key);
            return null;
        }

        return item.data;
    }

    setToCache(key, data, ttl = null) {
        const expiry = Date.now() + (ttl || this.cacheTTL);
        this.cache.set(key, { data, expiry });
        
        // محدود کردن اندازه کش
        if (this.cache.size > 100) {
            const firstKey = this.cache.keys().next().value;
            this.cache.delete(firstKey);
        }
    }

    clearCache() {
        this.cache.clear();
        console.log('🧹 AI Client cache cleared');
    }

    getCacheStats() {
        return {
            size: this.cache.size,
            keys: Array.from(this.cache.keys())
        };
    }

    // ===== آمار و مانیتورینگ پیشرفته =====
    updateUsageStats(success, responseTime) {
        this.usageStats.totalRequests++;
        
        if (success) {
            this.usageStats.successfulRequests++;
        } else {
            this.usageStats.failedRequests++;
        }

        // محاسبه میانگین زمان پاسخ
        if (responseTime > 0) {
            const currentAvg = this.usageStats.averageResponseTime;
            const totalSuccess = this.usageStats.successfulRequests;
            
            this.usageStats.averageResponseTime = 
                ((currentAvg * (totalSuccess - 1)) + responseTime) / totalSuccess;
        }

        this.usageStats.lastRequestTime = new Date().toISOString();
    }

    getUsageStats() {
        const successRate = this.usageStats.totalRequests > 0 ? 
            (this.usageStats.successfulRequests / this.usageStats.totalRequests) * 100 : 0;

        return {
            ...this.usageStats,
            successRate: Math.round(successRate * 100) / 100 + '%',
            averageResponseTime: Math.round(this.usageStats.averageResponseTime) + 'ms',
            cacheSize: this.cache.size,
            cacheHitRate: this.calculateCacheHitRate()
        };
    }

    calculateCacheHitRate() {
        const totalRequests = this.usageStats.totalRequests;
        if (totalRequests === 0) return '0%';
        
        const cacheHits = totalRequests - this.usageStats.successfulRequests - this.usageStats.failedRequests;
        const hitRate = (cacheHits / totalRequests) * 100;
        return Math.round(hitRate * 100) / 100 + '%';
    }

    // ===== مدیریت وضعیت و تاریخچه =====
    getStatus() {
        const aiStatus = {
            initialized: this.isInitialized,
            models: this.models,
            historyCount: this.analysisHistory.length,
            lastAnalysis: this.analysisHistory[this.analysisHistory.length - 1] || null,
            apiBase: this.apiBase,
            usageStats: this.getUsageStats(),
            cacheStats: this.getCacheStats(),
            performance: this.getPerformanceStats()
        };

        console.log('📊 AI Client Status:', aiStatus);
        return aiStatus;
    }

    getAnalysisHistory(symbol = null, type = null, limit = 50) {
        let history = this.analysisHistory;
        
        if (symbol) {
            history = history.filter(item => item.symbol === symbol);
        }
        
        if (type) {
            history = history.filter(item => item.type === type);
        }
        
        // مرتب‌سازی بر اساس زمان (جدیدترین اول)
        history.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
        
        return history.slice(0, limit);
    }

    getSymbolAnalysis(symbol) {
        const analyses = this.getAnalysisHistory(symbol);
        const technical = analyses.filter(a => a.type === 'technical');
        const sentiment = analyses.filter(a => a.type === 'sentiment');
        const prediction = analyses.filter(a => a.type === 'prediction');

        return {
            symbol,
            technical: technical[0] || null,
            sentiment: sentiment[0] || null,
            prediction: prediction[0] || null,
            totalAnalyses: analyses.length,
            firstAnalysis: analyses[analyses.length - 1] || null,
            lastAnalysis: analyses[0] || null
        };
    }

    clearHistory() {
        const count = this.analysisHistory.length;
        this.analysisHistory = [];
        console.log(`✅ AI analysis history cleared (${count} records)`);
        return count;
    }

    getPerformanceStats() {
        const technicalCount = this.analysisHistory.filter(item => item.type === 'technical').length;
        const sentimentCount = this.analysisHistory.filter(item => item.type === 'sentiment').length;
        const predictionCount = this.analysisHistory.filter(item => item.type === 'prediction').length;
        
        const totalResponseTime = this.analysisHistory.reduce((sum, item) => sum + (item.responseTime || 0), 0);
        const avgResponseTime = this.analysisHistory.length > 0 ? totalResponseTime / this.analysisHistory.length : 0;

        const uniqueSymbols = [...new Set(this.analysisHistory.map(item => item.symbol))];

        return {
            total_analyses: this.analysisHistory.length,
            technical_analyses: technicalCount,
            sentiment_analyses: sentimentCount,
            predictions: predictionCount,
            unique_symbols: uniqueSymbols.length,
            symbols_list: uniqueSymbols,
            average_response_time: Math.round(avgResponseTime) + 'ms',
            success_rate: this.getUsageStats().successRate,
            cache_performance: this.getUsageStats().cacheHitRate
        };
    }

    // ===== ابزار کمکی =====
    stringToHash(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;
        }
        return Math.abs(hash);
    }

    // ===== متدهای کمکی برای توسعه =====
    simulateAnalysis(symbol, type = 'technical') {
        console.log(`🎭 Simulating ${type} analysis for ${symbol}`);
        
        if (type === 'technical') {
            return this.fallbackTechnicalAnalysis(null, symbol);
        } else if (type === 'sentiment') {
            return this.fallbackSentimentAnalysis(symbol);
        } else if (type === 'prediction') {
            return this.fallbackPrediction(symbol, '1d');
        }
    }

    // تست اتصال پیشرفته
    async testConnection() {
        try {
            const startTime = Date.now();
            const response = await fetch('/api/ai/status');
            const responseTime = Date.now() - startTime;

            if (response.ok) {
                const status = await response.json();
                return {
                    connected: true,
                    responseTime: responseTime + 'ms',
                    status: status.status,
                    modules: status.modules,
                    performance: status.performance,
                    details: 'AI backend is operational'
                };
            } else {
                return {
                    connected: false,
                    error: `HTTP ${response.status}`,
                    responseTime: responseTime + 'ms',
                    details: 'AI backend responded with error'
                };
            }
        } catch (error) {
            return {
                connected: false,
                error: error.message,
                responseTime: 0,
                details: 'Failed to connect to AI backend'
            };
        }
    }

    // سلامت سیستم
    async healthCheck() {
        const connectionTest = await this.testConnection();
        const stats = this.getUsageStats();
        const performance = this.getPerformanceStats();
        
        return {
            connection: connectionTest,
            usage: stats,
            performance: performance,
            cache: this.getCacheStats(),
            overall: connectionTest.connected && this.isInitialized ? 'healthy' : 'unhealthy'
        };
    }

    // ریست سیستم
    async reset() {
        this.clearCache();
        this.clearHistory();
        this.usageStats = {
            totalRequests: 0,
            successfulRequests: 0,
            failedRequests: 0,
            averageResponseTime: 0,
            lastRequestTime: null
        };
        
        console.log('🔄 AI Client reset completed');
        return this.initialize();
    }
}

// ایجاد نمونه جهانی برای دسترسی آسان
if (typeof window !== 'undefined') {
    window.AIClient = AIClient;
    window.aiClient = new AIClient();
}
