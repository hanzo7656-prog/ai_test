// کلاینت هوش مصنوعی VortexAI - سازگار با روت‌های جدید
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
    }

    async initialize() {
        try {
            // تست اتصال به AI backend از طریق روت سلامت
            const response = await fetch('/api/status');
            const status = await response.json();
            
            if (status.status === 'operational' && status.services.ai_engine) {
                this.isInitialized = true;
                this.models = status.ai_capabilities || {};
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

    async analyzeTechnical(symbol, data = null) {
        try {
            if (!this.isInitialized) {
                await this.initialize();
            }

            console.log(`🧠 تحلیل تکنیکال AI برای ${symbol}`);

            // اگر داده‌ای ارائه نشده، از سرور بگیر
            let rawData = data;
            if (!rawData) {
                const rawResponse = await fetch(`/api/raw/${symbol}`);
                const result = await rawResponse.json();
                rawData = result.data;
            }

            // ارسال درخواست به AI backend با روت جدید
            const response = await fetch(`${this.apiBase}/analyze/${symbol}?analysis_type=technical`);

            if (!response.ok) {
                throw new Error(`AI analysis failed: ${response.status}`);
            }

            const result = await response.json();
            
            // ذخیره در تاریخچه
            this.analysisHistory.push({
                symbol: symbol,
                analysis: result,
                timestamp: new Date().toISOString(),
                type: 'technical'
            });

            return result;

        } catch (error) {
            console.error(`خطا در تحلیل AI برای ${symbol}:`, error);
            // Fallback به تحلیل ساده
            return this.fallbackTechnicalAnalysis(data, symbol);
        }
    }

    async analyzeSentiment(symbol) {
        try {
            console.log(`😊 تحلیل احساسات AI برای ${symbol}`);

            const response = await fetch(`${this.apiBase}/analyze/${symbol}?analysis_type=sentiment`);

            if (!response.ok) {
                throw new Error(`Sentiment analysis failed: ${response.status}`);
            }

            const result = await response.json();
            
            this.analysisHistory.push({
                symbol: symbol,
                analysis: result,
                timestamp: new Date().toISOString(),
                type: 'sentiment'
            });

            return result;

        } catch (error) {
            console.error(`خطا در تحلیل احساسات برای ${symbol}:`, error);
            return this.fallbackSentimentAnalysis(symbol);
        }
    }

    async getPrediction(symbol, period = '1d') {
        try {
            console.log(`🔮 پیش‌بینی AI برای ${symbol} (${period})`);

            const response = await fetch(`${this.apiBase}/analyze/${symbol}?analysis_type=prediction&period=${period}`);

            if (!response.ok) {
                throw new Error(`Prediction failed: ${response.status}`);
            }

            const result = await response.json();
            
            this.analysisHistory.push({
                symbol: symbol,
                analysis: result,
                timestamp: new Date().toISOString(),
                type: 'prediction',
                period: period
            });

            return result;

        } catch (error) {
            console.error(`خطا در دریافت پیش‌بینی برای ${symbol}:`, error);
            return this.fallbackPrediction(symbol, period);
        }
    }

    // متدهای Fallback
    fallbackTechnicalAnalysis(data, symbol) {
        const price = data?.market_data?.price || data?.price || 0;
        const change = data?.market_data?.priceChange1d || data?.change || 0;
        const volume = data?.market_data?.volume || data?.volume || 0;
        
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
                volume_impact: volume > 1000000000 ? 'بالا' : 'عادی',
                price_change: change
            },
            summary: this.generateSummary(signal, confidence, rsi, change),
            timestamp: new Date().toISOString(),
            source: 'fallback'
        };

        console.log(`🔶 استفاده از تحلیل fallback برای ${symbol}:`, analysis);
        return analysis;
    }

    fallbackSentimentAnalysis(symbol) {
        const sentiment = {
            symbol: symbol,
            score: 0.3 + Math.random() * 0.4, // 0.3-0.7
            trend: Math.random() > 0.5 ? 'positive' : 'negative',
            confidence: 0.4 + Math.random() * 0.3,
            indicators: {
                social_volume: 'medium',
                news_sentiment: 'neutral',
                market_mood: Math.random() > 0.5 ? 'bullish' : 'bearish'
            },
            summary: 'تحلیل احساسات در دسترس نیست',
            timestamp: new Date().toISOString(),
            source: 'fallback'
        };

        return sentiment;
    }

    fallbackPrediction(symbol, period) {
        const basePrice = 1000 + (this.stringToHash(symbol) % 50000);
        const volatility = 0.02; // 2% نوسان
        
        return {
            symbol: symbol,
            period: period,
            prediction: {
                predicted_price: basePrice * (1 + (Math.random() - 0.5) * volatility),
                confidence: 0.3 + Math.random() * 0.4,
                direction: Math.random() > 0.5 ? 'up' : 'down',
                volatility: volatility,
                time_frame: period
            },
            timestamp: new Date().toISOString(),
            source: 'fallback',
            disclaimer: 'پیش‌بینی بر اساس داده‌های محدود'
        };
    }

    // ابزارهای تحلیل
    calculateSimpleRSI(change) {
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

        if (Math.abs(change) > 10) {
            parts.push('نوسان شدید');
        }

        return parts.join(' • ');
    }

    // مدیریت وضعیت و تاریخچه
    getStatus() {
        return {
            initialized: this.isInitialized,
            technical: this.models.technical,
            sentiment: this.models.sentiment,
            predictive: this.models.predictive,
            historyCount: this.analysisHistory.length,
            lastAnalysis: this.analysisHistory[this.analysisHistory.length - 1] || null,
            apiBase: this.apiBase
        };
    }

    getAnalysisHistory(symbol = null, type = null) {
        let history = this.analysisHistory;
        
        if (symbol) {
            history = history.filter(item => item.symbol === symbol);
        }
        
        if (type) {
            history = history.filter(item => item.type === type);
        }
        
        return history;
    }

    clearHistory() {
        this.analysisHistory = [];
        console.log('✅ AI analysis history cleared');
    }

    getPerformanceStats() {
        const technicalCount = this.analysisHistory.filter(item => item.type === 'technical').length;
        const sentimentCount = this.analysisHistory.filter(item => item.type === 'sentiment').length;
        const predictionCount = this.analysisHistory.filter(item => item.type === 'prediction').length;
        
        return {
            total_analyses: this.analysisHistory.length,
            technical_analyses: technicalCount,
            sentiment_analyses: sentimentCount,
            predictions: predictionCount,
            unique_symbols: [...new Set(this.analysisHistory.map(item => item.symbol))].length
        };
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
