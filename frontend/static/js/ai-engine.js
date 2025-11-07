// ===== سیستم هوش مصنوعی پایه =====
class SimpleAI {
    constructor() {
        this.isInitialized = false;
        this.models = {
            technical: null,
            sentiment: null,
            predictive: null
        };
        this.history = [];
    }

    async initialize() {
        try {
            // بارگذاری مدل‌های پایه
            await this.loadTechnicalModel();
            await this.loadSentimentModel();
            await this.loadPredictiveModel();
            
            this.isInitialized = true;
            return true;
        } catch (error) {
            console.error('AI Initialization error:', error);
            return false;
        }
    }

    async loadTechnicalModel() {
        // مدل تحلیل تکنیکال ساده
        this.models.technical = {
            name: 'تحلیل‌گر تکنیکال',
            version: '1.0',
            ready: true,
            indicators: ['RSI', 'MACD', 'MovingAverage', 'SupportResistance']
        };
        await this.delay(500);
    }

    async loadSentimentModel() {
        // مدل تحلیل احساسات ساده
        this.models.sentiment = {
            name: 'تحلیل‌گر احساسات',
            version: '1.0',
            ready: true,
            sources: ['PriceAction', 'VolumeAnalysis', 'MarketRank']
        };
        await this.delay(300);
    }

    async loadPredictiveModel() {
        // مدل پیش‌بینی ساده
        this.models.predictive = {
            name: 'پیش‌بین قیمت',
            version: '1.0',
            ready: true,
            features: ['HistoricalPatterns', 'MarketCycles', 'VolatilityAnalysis']
        };
        await this.delay(400);
    }

    analyzeTechnical(coinData) {
        const analysis = {
            signal: 'HOLD',
            confidence: 0.5,
            indicators: [],
            summary: ''
        };

        // تحلیل بر اساس RSI ساده
        const rsi = this.calculateRSI(coinData);
        if (rsi < 30) {
            analysis.signal = 'BUY';
            analysis.confidence += 0.2;
            analysis.indicators.push(`RSI: ${rsi.toFixed(1)} (اشباع فروش)`);
        } else if (rsi > 70) {
            analysis.signal = 'SELL';
            analysis.confidence += 0.2;
            analysis.indicators.push(`RSI: ${rsi.toFixed(1)} (اشباع خرید)`);
        }

        // تحلیل روند قیمت
        if (coinData.change > 5) {
            analysis.signal = analysis.signal === 'SELL' ? 'HOLD' : 'BUY';
            analysis.confidence += 0.15;
            analysis.indicators.push(`روند: صعودی (${coinData.change.toFixed(1)}%)`);
        } else if (coinData.change < -5) {
            analysis.signal = analysis.signal === 'BUY' ? 'HOLD' : 'SELL';
            analysis.confidence += 0.15;
            analysis.indicators.push(`روند: نزولی (${coinData.change.toFixed(1)}%)`);
        }

        // تحلیل حجم
        if (coinData.volume > 500000000) { // حجم بالا
            analysis.confidence += 0.1;
            analysis.indicators.push('حجم: بالا');
        }

        // تحلیل رتبه بازار
        if (coinData.rank && coinData.rank <= 10) {
            analysis.confidence += 0.1;
            analysis.indicators.push('رتبه: برتر');
        }

        // محدود کردن confidence
        analysis.confidence = Math.max(0.1, Math.min(0.95, analysis.confidence));

        // ارتقا سیگنال بر اساس confidence
        if (analysis.confidence > 0.7 && analysis.signal === 'BUY') {
            analysis.signal = 'STRONG_BUY';
        } else if (analysis.confidence > 0.7 && analysis.signal === 'SELL') {
            analysis.signal = 'STRONG_SELL';
        }

        analysis.summary = analysis.indicators.join(' • ') || 'داده کافی نیست';

        // ذخیره در تاریخچه
        this.history.push({
            symbol: coinData.name,
            analysis,
            timestamp: new Date().toISOString()
        });

        return analysis;
    }

    calculateRSI(coinData) {
        // شبیه‌سازی RSI ساده بر اساس تغییرات قیمت
        const change = coinData.change || 0;
        return Math.min(100, Math.max(0, 50 + (change * 2)));
    }

    getStatus() {
        return {
            initialized: this.isInitialized,
            technical: this.models.technical,
            sentiment: this.models.sentiment,
            predictive: this.models.predictive,
            historyCount: this.history.length
        };
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}
