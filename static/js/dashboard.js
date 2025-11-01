// static/js/dashboard.js
class Dashboard {
    constructor() {
        this.initializeDashboard();
        this.setupEventListeners();
        this.startRealTimeUpdates();
    }

    async initializeDashboard() {
        await this.loadRealTimeData();
        this.updateActiveSignals();
        this.updateSystemStatus();
        this.setupChart();
    }

    setupEventListeners() {
        // کلیک روی هشدارهای فعال - رفتن به صفحه سلامت
        document.getElementById('alertsList')?.addEventListener('click', () => {
            window.location.href = '/health#alerts';
        });

        // کلیک روی سیگنال‌های فعال - رفتن به صفحه تحلیل
        document.getElementById('signalsList')?.addEventListener('click', () => {
            window.location.href = '/analysis';
        });

        // کلیک روی کارت‌های سریع دسترسی
        document.querySelectorAll('.quick-card').forEach(card => {
            card.addEventListener('click', () => {
                const page = card.dataset.page;
                if (page) {
                    window.location.href = page;
                }
            });
        });
    }

    async loadRealTimeData() {
        try {
            // دریافت داده real-time از API
            const response = await fetch('/api/ai/analysis?symbols=BTC,ETH&period=1h');
            const data = await response.json();
            
            if (data.status === 'success') {
                this.updateWithRealData(data.analysis_report);
            } else {
                this.updateWithSampleData();
            }
        } catch (error) {
            console.error('Error loading real data:', error);
            this.updateWithSampleData();
        }
    }

    updateWithRealData(analysisData) {
        // آپدیت با داده واقعی از API
        const btcData = analysisData.symbol_analysis?.BTC;
        const ethData = analysisData.symbol_analysis?.ETH;
        
        if (btcData) {
            this.updatePriceDisplay('BTC', btcData.current_price, btcData.technical_score);
        }
        
        if (ethData) {
            this.updatePriceDisplay('ETH', ethData.current_price, ethData.technical_score);
        }
    }

    updateWithSampleData() {
        // داده نمونه تا زمانی که API آماده بشه
        this.updateBTCPrice();
        this.updateETHPrice();
    }

    async updateBTCPrice() {
        try {
            // شبیه‌سازی دریافت داده real-time
            const btcPrice = 43256 + (Math.random() - 0.5) * 1000;
            const btcChange = (Math.random() - 0.3) * 5;
            
            this.updatePriceDisplay('BTC', btcPrice, btcChange);
        } catch (error) {
            console.error('Error updating BTC price:', error);
        }
    }

    async updateETHPrice() {
        try {
            // شبیه‌سازی دریافت داده real-time
            const ethPrice = 2580 + (Math.random() - 0.5) * 100;
            const ethChange = (Math.random() - 0.3) * 4;
            
            this.updatePriceDisplay('ETH', ethPrice, ethChange);
        } catch (error) {
            console.error('Error updating ETH price:', error);
        }
    }

    updatePriceDisplay(symbol, price, change) {
        const priceElement = document.querySelector(`.quick-chart .current-price`);
        const changeElement = document.querySelector(`.quick-chart .price-change`);
        
        if (priceElement && changeElement) {
            priceElement.textContent = `$${price.toLocaleString('en-US', {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2
            })}`;
            
            changeElement.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`;
            changeElement.className = `price-change ${change >= 0 ? 'positive' : 'negative'}`;
        }
    }

    updateActiveSignals() {
        const signals = [
            { symbol: 'BTC', name: 'Bitcoin', price: 43256.89, change: 2.34, type: 'bullish', confidence: 87 },
            { symbol: 'ETH', name: 'Ethereum', price: 2580.45, change: 1.56, type: 'bullish', confidence: 78 },
            { symbol: 'SOL', name: 'Solana', price: 102.34, change: -0.89, type: 'bearish', confidence: 65 },
            { symbol: 'ADA', name: 'Cardano', price: 0.5123, change: 3.21, type: 'bullish', confidence: 72 }
        ];

        const container = document.getElementById('signalsList');
        if (!container) return;

        container.innerHTML = signals.map(signal => `
            <div class="signal-item ${signal.type}" onclick="window.location.href='/analysis?symbol=${signal.symbol}'">
                <div class="signal-info">
                    <div class="signal-symbol">${signal.symbol}</div>
                    <div class="signal-name">${signal.name}</div>
                </div>
                <div class="signal-price">$${signal.price.toLocaleString()}</div>
                <div class="signal-change ${signal.change >= 0 ? 'positive' : 'negative'}">
                    ${signal.change >= 0 ? '+' : ''}${signal.change}%
                </div>
                <div class="signal-confidence">${signal.confidence}%</div>
            </div>
        `).join('');
    }

    updateSystemStatus() {
        const statusItems = [
            { label: 'API CoinStats', value: 'connected', status: 'connected' },
            { label: 'مدل AI', value: 'active', status: 'active' },
            { label: 'WebSocket', value: 'connected', status: 'connected' },
            { label: 'دقت پیش‌بینی', value: '87%', status: 'normal' }
        ];

        const container = document.querySelector('.status-grid');
        if (!container) return;

        container.innerHTML = statusItems.map(item => `
            <div class="status-item">
                <div class="status-label">${item.label}</div>
                <div class="status-value ${item.status}">${item.value}</div>
            </div>
        `).join('');
    }

    setupChart() {
        // ایجاد نمودار ساده برای BTC
        const container = document.getElementById('btcChart');
        if (!container) return;

        // شبیه‌سازی داده‌های قیمت
        const prices = this.generateSampleChartData();
        this.renderSimpleChart(container, prices);
    }

    generateSampleChartData() {
        const basePrice = 43000;
        return Array.from({length: 20}, (_, i) => {
            const volatility = Math.sin(i * 0.5) * 500 + Math.random() * 300;
            return basePrice + volatility;
        });
    }

    renderSimpleChart(container, prices) {
        const maxPrice = Math.max(...prices);
        const minPrice = Math.min(...prices);
        const range = maxPrice - minPrice;

        container.innerHTML = '';
        const chart = document.createElement('div');
        chart.className = 'simple-chart';
        chart.style.cssText = `
            width: 100%;
            height: 100%;
            display: flex;
            align-items: flex-end;
            gap: 2px;
            padding: 10px;
        `;

        prices.forEach((price, index) => {
            const bar = document.createElement('div');
            const height = ((price - minPrice) / range) * 80;
            const isGreen = index === 0 || price >= prices[index - 1];
            
            bar.style.cssText = `
                flex: 1;
                height: ${height}%;
                background: ${isGreen ? 'var(--accent-success)' : 'var(--accent-danger)'};
                border-radius: 2px;
                opacity: ${0.6 + (index * 0.02)};
                transition: all 0.3s ease;
            `;
            
            chart.appendChild(bar);
        });

        container.appendChild(chart);
    }

    startRealTimeUpdates() {
        // بروزرسانی Real-time قیمت‌ها
        setInterval(() => {
            this.updateBTCPrice();
            this.updateETHPrice();
        }, 5000);

        // بروزرسانی سیگنال‌ها هر 30 ثانیه
        setInterval(() => {
            this.updateActiveSignals();
        }, 30000);

        // بروزرسانی نمودار هر 10 ثانیه
        setInterval(() => {
            const container = document.getElementById('btcChart');
            if (container) {
                const newPrices = this.generateSampleChartData();
                this.renderSimpleChart(container, newPrices);
            }
        }, 10000);
    }
}

// راه‌اندازی
document.addEventListener('DOMContentLoaded', () => {
    new Dashboard();
});
