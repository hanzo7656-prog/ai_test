// کلاینت هوش مصنوعی VortexAI - ارتباط با backend پایتون
class AIClient {
    constructor() {
        this.isInitialized = false;
        this.models = {
            technical: null,
            sentiment: null,
            predictive: null
        };
        this.analysisHistory = [];
    }

    async initialize() {
        try {
            // تست اتصال به AI backend
            const response = await fetch('/api/ai/status');
            const status = await response.json();
            
            if (status.status === 'operational') {
                this.isInitialized = true;
                this.models = status.models || {};
                console.log('✅ AI Client initialized successfully');
                return true;
            } else {
                throw new Error('AI backend not operational');
            }
        } catch (error) {
            console.error('❌ AI Client initialization failed:', error);
            // Fallback به حالت شبیه‌سازی برای توسعه
            return this.initializeFallback();
        }
    }

    async initializeFallback() {
        // شبیه‌سازی برای زمانی که AI backend در دسترس نیست
        this.models = {
            technical: { name: 'تحلیل‌گر تکنیکال', ready: true, version: '1.0' },
            sentiment: { name: 'تحلیل‌گر احساسات', ready: false, version: '1.0' },
            predictive: { name: 'پیش‌بین قیمت', ready: false, version: '1.0' }
        };
        this.isInitialized = true;
        console.log('🔶 AI Client running in fallback mode');
        return true;
    }

    async analyzeTechnical(symbol, data) {
        try {
            if (!this.isInitialized) {
                throw new Error('AI client not initialized');
            }

            console.log(`🧠 تحلیل تکنیکال AI برای ${symbol}`);

            // ارسال درخواست به AI backend
            const response = await fetch('/api/ai/analyze/technical', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    symbol: symbol,
                    data: data,
                    timestamp: new Date().toISOString()
                })
            });

            if (!response.ok) {
                throw new Error(`AI analysis failed: ${response.status}`);
            }

            const result = await response.json();
            
            // ذخیره در تاریخچه
            this.analysisHistory.push({
                symbol: symbol,
                analysis: result,
                timestamp: new Date().toISOString()
            });

            return result;

        } catch (error) {
            console.error(`خطا در تحلیل AI برای ${symbol}:`, error);
            // Fallback به تحلیل ساده
            return this.fallbackTechnicalAnalysis(data, symbol);
        }
    }

    fallbackTechnicalAnalysis(data, symbol) {
        // تحلیل تکنیکال ساده به عنوان fallback
        const price = data.price || 0;
        const change = data.change || 0;
        const volume = data.volume || 0;
        
        // محاسبه RSI ساده
        const rsi = this.calculateSimpleRSI(change);
        
        // تحلیل روند
        let signal = 'HOLD';
        let confidence = 0.5;

        if (rsi < 30 && change > 0) {
            signal = 'STRONG_BUY';
            confidence = 0.8;
        } else if (rsi < 40 && change > 0) {
            signal = 'BUY';
            confidence = 0.6;
        } else if (rsi > 70 && change < 0) {
            signal = 'STRONG_SELL';
            confidence = 0.8;
        } else if (rsi > 60 && change < 0) {
            signal = 'SELL';
            confidence = 0.6;
        }

        // افزایش confidence بر اساس حجم
        if (volume > 1000000000) {
            confidence = Math.min(0.95, confidence + 0.15);
        }

        const analysis = {
            signal: signal,
            confidence: confidence,
            indicators: {
                rsi: rsi,
                trend: change > 0 ? 'صعودی' : 'نزولی',
                volume_impact: volume > 1000000000 ? 'بالا' : 'عادی'
            },
            summary: this.generateSummary(signal, confidence, rsi, change),
            timestamp: new Date().toISOString(),
            source: 'fallback'
        };

        console.log(`🔶 استفاده از تحلیل fallback برای ${symbol}:`, analysis);
        return analysis;
    }

    calculateSimpleRSI(change) {
        // شبیه‌سازی RSI ساده
        return Math.min(100, Math.max(0, 50 + (change * 2)));
    }

    generateSummary(signal, confidence, rsi, change) {
        const parts = [];
        
        if (signal.includes('BUY')) {
            parts.push('سیگنال خرید');
        } else if (signal.includes('SELL')) {
            parts.push('سیگنال فروش');
        } else {
            parts.push('سیگنال نگهداری');
        }

        if (confidence > 0.7) {
            parts.push('اعتماد بالا');
        } else if (confidence > 0.5) {
            parts.push('اعتماد متوسط');
        } else {
            parts.push('اعتماد پایین');
        }

        if (rsi < 30) {
            parts.push('اشباع فروش');
        } else if (rsi > 70) {
            parts.push('اشباع خرید');
        }

        return parts.join(' • ');
    }

    async getAIPrediction(symbol, period = '1d') {
        try {
            const response = await fetch(`/api/ai/predict/${symbol}?period=${period}`);
            
            if (!response.ok) {
                throw new Error(`Prediction failed: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error(`خطا در دریافت پیش‌بینی برای ${symbol}:`, error);
            return this.fallbackPrediction(symbol, period);
        }
    }

    fallbackPrediction(symbol, period) {
        // پیش‌بینی ساده fallback
        const basePrice = 1000 + (this.stringToHash(symbol) % 50000);
        const volatility = 0.02; // 2% نوسان
        
        return {
            symbol: symbol,
            period: period,
            prediction: {
                price: basePrice * (1 + (Math.random() - 0.5) * volatility),
                confidence: 0.3 + Math.random() * 0.4,
                direction: Math.random() > 0.5 ? 'up' : 'down',
                volatility: volatility
            },
            timestamp: new Date().toISOString(),
            source: 'fallback'
        };
    }

    async getMarketSentiment(symbols = ['bitcoin', 'ethereum']) {
        try {
            const response = await fetch('/api/ai/sentiment', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ symbols: symbols })
            });
            
            if (!response.ok) {
                throw new Error(`Sentiment analysis failed: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('خطا در تحلیل احساسات:', error);
            return this.fallbackSentiment(symbols);
        }
    }

    fallbackSentiment(symbols) {
        // تحلیل احساسات fallback
        const sentiment = {};
        
        symbols.forEach(symbol => {
            sentiment[symbol] = {
                score: 0.3 + Math.random() * 0.4, // 0.3-0.7
                trend: Math.random() > 0.5 ? 'positive' : 'negative',
                volume: 'normal',
                timestamp: new Date().toISOString()
            };
        });

        return {
            sentiments: sentiment,
            overall_score: 0.5,
            market_mood: 'neutral',
            source: 'fallback'
        };
    }

    getStatus() {
        return {
            initialized: this.isInitialized,
            technical: this.models.technical,
            sentiment: this.models.sentiment,
            predictive: this.models.predictive,
            historyCount: this.analysisHistory.length,
            lastAnalysis: this.analysisHistory[this.analysisHistory.length - 1] || null
        };
    }

    getAnalysisHistory(symbol = null) {
        if (symbol) {
            return this.analysisHistory.filter(item => item.symbol === symbol);
        }
        return this.analysisHistory;
    }

    clearHistory() {
        this.analysisHistory = [];
        console.log('✅ AI analysis history cleared');
    }

    // ابزار کمکی
    stringToHash(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;
        }
        return Math.abs(hash);
    }
}
