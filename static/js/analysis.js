// static/js/analysis.js
class TechnicalAnalysis {
    constructor() {
        this.currentSymbol = 'BTCUSDT';
        this.currentTimeframe = '1h';
        this.initializeChart();
        this.setupEventListeners();
        this.loadAnalysisData();
        this.startRealTimeUpdates();
    }

    initializeChart() {
        // ایجاد نمودار ساده با CSS
        this.createSimpleChart();
    }

    createSimpleChart() {
        const container = document.getElementById('mainChart');
        if (!container) return;

        // شبیه‌سازی داده‌های قیمت
        const prices = this.generateSampleData();
        
        container.innerHTML = '';
        const chart = document.createElement('div');
        chart.className = 'simple-price-chart';
        
        // ایجاد خط نمودار
        const svg = this.createSVGChart(prices);
        chart.appendChild(svg);
        
        container.appendChild(chart);
    }

    createSVGChart(prices) {
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('viewBox', '0 0 400 200');
        svg.setAttribute('width', '100%');
        svg.setAttribute('height', '100%');

        // محاسبه نقاط
        const points = prices.map((price, index) => {
            const x = (index / (prices.length - 1)) * 400;
            const y = 200 - ((price - Math.min(...prices)) / (Math.max(...prices) - Math.min(...prices))) * 180;
            return `${x},${y}`;
        }).join(' ');

        // خط نمودار
        const polyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
        polyline.setAttribute('points', points);
        polyline.setAttribute('fill', 'none');
        polyline.setAttribute('stroke', '#13bcff');
        polyline.setAttribute('stroke-width', '2');
        svg.appendChild(polyline);

        // نقاط کلیدی
        [0, prices.length - 1].forEach(index => {
            const x = (index / (prices.length - 1)) * 400;
            const y = 200 - ((prices[index] - Math.min(...prices)) / (Math.max(...prices) - Math.min(...prices))) * 180;
            
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', x);
            circle.setAttribute('cy', y);
            circle.setAttribute('r', '3');
            circle.setAttribute('fill', '#13bcff');
            svg.appendChild(circle);

            // متن قیمت
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', x);
            text.setAttribute('y', y - 10);
            text.setAttribute('text-anchor', index === 0 ? 'start' : 'end');
            text.setAttribute('fill', '#ffffff');
            text.setAttribute('font-size', '10');
            text.textContent = `$${prices[index].toLocaleString()}`;
            svg.appendChild(text);
        });

        return svg;
    }

    generateSampleData() {
        const basePrice = 43000;
        return Array.from({length: 50}, (_, i) => {
            const volatility = Math.sin(i * 0.3) * 500 + Math.random() * 300;
            return basePrice + volatility;
        });
    }

    setupEventListeners() {
        // تغییر نماد
        document.getElementById('symbolSelect')?.addEventListener('change', (e) => {
            this.currentSymbol = e.target.value;
            this.updateAnalysis();
        });

        // تغییر تایم‌فریم
        document.querySelectorAll('.timeframe-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.timeframe-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                this.currentTimeframe = e.target.dataset.tf;
                this.updateAnalysis();
            });
        });

        // ابزارهای نمودار
        document.querySelectorAll('.tool-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const tool = e.target.dataset.tool;
                this.handleChartTool(tool);
            });
        });

        // بروزرسانی اندیکاتورها
        document.getElementById('refreshIndicators')?.addEventListener('click', () => {
            this.refreshIndicators();
        });

        // toggle تحلیل عمیق
        document.getElementById('deepAnalysisToggle')?.addEventListener('change', (e) => {
            this.toggleDeepAnalysis(e.target.checked);
        });
    }

    handleChartTool(tool) {
        const tools = {
            draw: 'ابزار رسم فعال شد',
            indicators: 'مدیریت اندیکاتورها',
            fullscreen: 'حالت تمام صفحه'
        };
        
        if (tools[tool]) {
            this.showNotification(tools[tool]);
        }
    }

    async loadAnalysisData() {
        // شبیه‌سازی بارگذاری داده‌ها
        await this.simulateLoading();
        this.updatePriceDisplay();
        this.updateIndicators();
        this.updateSentiment();
    }

    simulateLoading() {
        return new Promise(resolve => setTimeout(resolve, 1000));
    }

    updatePriceDisplay() {
        const price = 43256.89 + (Math.random() - 0.5) * 1000;
        const change = (Math.random() - 0.3) * 5;
        
        document.getElementById('currentPrice').textContent = `$${price.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
        
        const changeElement = document.getElementById('priceChange');
        changeElement.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`;
        changeElement.className = `change ${change >= 0 ? 'positive' : 'negative'}`;
    }

    updateIndicators() {
        // شبیه‌سازی بروزرسانی اندیکاتورها
        const indicators = {
            rsi: { value: (40 + Math.random() * 40).toFixed(1), status: 'neutral' },
            macd: { value: (Math.random() - 0.5) * 5, status: 'bullish' },
            ema: 42980 + (Math.random() - 0.5) * 1000,
            volume: (1.5 + Math.random() * 2).toFixed(1) + 'B'
        };

        // آپدیت مقادیر در صفحه
        this.updateIndicatorElement('RSI', indicators.rsi.value, indicators.rsi.status);
        this.updateIndicatorElement('MACD', indicators.macd.value.toFixed(2), indicators.macd.status);
        this.updateIndicatorElement('EMA 20', `$${Math.round(indicators.ema).toLocaleString()}`, 'neutral');
        this.updateIndicatorElement('Volume', indicators.volume, 'neutral');
    }

    updateIndicatorElement(name, value, status) {
        const items = document.querySelectorAll('.indicator-item');
        items.forEach(item => {
            if (item.querySelector('.indicator-name').textContent === name) {
                const valueElement = item.querySelector('.indicator-value');
                valueElement.textContent = value;
                valueElement.className = `indicator-value ${status}`;
            }
        });
    }

    updateSentiment() {
        const fearGreed = 30 + Math.random() * 70;
        const volatility = 50 + Math.random() * 50;
        
        // آپدیت مترهای احساسات
        document.querySelectorAll('.meter-fill')[0].style.width = `${fearGreed}%`;
        document.querySelectorAll('.meter-value')[0].textContent = `${Math.round(fearGreed)} - ${this.getSentimentText(fearGreed)}`;
        
        document.querySelectorAll('.meter-fill')[1].style.width = `${volatility}%`;
        document.querySelectorAll('.meter-value')[1].textContent = `${Math.round(volatility)}% - ${volatility > 70 ? 'بالا' : 'متوسط'}`;
        
        // آپدیت امتیاز احساسات
        const sentimentScore = document.querySelector('.sentiment-score');
        sentimentScore.textContent = Math.round(fearGreed);
        sentimentScore.className = `sentiment-score ${fearGreed > 60 ? 'positive' : fearGreed > 40 ? 'neutral' : 'negative'}`;
    }

    getSentimentText(score) {
        if (score >= 70) return 'طمع';
        if (score >= 60) return 'امیدوار';
        if (score >= 40) return 'خنثی';
        if (score >= 30) return 'ترس';
        return 'ترس شدید';
    }

    refreshIndicators() {
        this.showNotification('اندیکاتورها بروزرسانی شدند');
        this.updateIndicators();
    }

    toggleDeepAnalysis(enabled) {
        const content = document.getElementById('deepAnalysisContent');
        if (content) {
            content.style.display = enabled ? 'block' : 'none';
        }
    }

    updateAnalysis() {
        this.showNotification(`تحلیل برای ${this.currentSymbol} (${this.currentTimeframe}) بروزرسانی شد`);
        this.initializeChart();
        this.loadAnalysisData();
    }

    startRealTimeUpdates() {
        // بروزرسانی Real-time قیمت
        setInterval(() => {
            this.updatePriceDisplay();
        }, 5000);

        // بروزرسانی اندیکاتورها هر 30 ثانیه
        setInterval(() => {
            this.updateIndicators();
        }, 30000);
    }

    showNotification(message) {
        // ایجاد نوتیفیکیشن موقت
        const notification = document.createElement('div');
        notification.className = 'analysis-notification';
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: var(--accent-primary);
            color: white;
            padding: 1rem 2rem;
            border-radius: 8px;
            z-index: 10000;
            animation: slideDown 0.3s ease;
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }
}

// راه‌اندازی
document.addEventListener('DOMContentLoaded', () => {
    new TechnicalAnalysis();
});
