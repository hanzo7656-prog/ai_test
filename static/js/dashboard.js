// static/js/dashboard.js - کاملاً هماهنگ با API های واقعی
class Dashboard {
    constructor() {
        this.systemStatus = {};
        this.marketData = {};
        this.activeAlerts = [];
        this.initializeDashboard();
    }

    async initializeDashboard() {
        await this.loadSystemStatus();
        await this.loadMarketData();
        await this.loadActiveAlerts();
        this.setupEventListeners();
        this.startRealTimeUpdates();
    }

    async loadSystemStatus() {
        try {
            console.log('🔄 دریافت وضعیت سیستم از API...');
            const response = await fetch('/api/system/health');
            
            if (!response.ok) {
                throw new Error(`API سلامت خطا: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('📊 وضعیت سیستم:', data);
            
            this.systemStatus = data;
            this.renderSystemStatus();
            
        } catch (error) {
            console.error('❌ خطا در دریافت وضعیت سیستم:', error);
            this.renderSystemStatusError();
        }
    }

    async loadMarketData() {
        try {
            console.log('🔄 دریافت داده‌های بازار از API اسکن...');
            
            const response = await fetch('/api/ai/scan/advanced', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    symbols: ["BTC", "ETH", "SOL", "ADA", "DOT", "LINK", "BNB", "XRP", "DOGE", "MATIC"],
                    conditions: {
                        min_confidence: 0.6,
                        max_change: 10
                    },
                    timeframe: "1h"
                })
            });

            if (!response.ok) {
                throw new Error(`API اسکن خطا: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('📊 داده‌های اسکن:', data);

            if (data.status === 'success' && data.scan_results) {
                this.marketData = data.scan_results;
                this.renderMarketData();
            } else {
                throw new Error('داده معتبر از API اسکن دریافت نشد');
            }

        } catch (error) {
            console.error('❌ خطا در دریافت داده بازار:', error);
            this.renderMarketDataError();
        }
    }

    async loadActiveAlerts() {
        try {
            console.log('🔄 دریافت هشدارهای فعال از API...');
            const response = await fetch('/api/system/alerts');
            
            if (response.ok) {
                const data = await response.json();
                this.activeAlerts = data.alerts || [];
                this.renderActiveAlerts();
            }
        } catch (error) {
            console.error('❌ خطا در دریافت هشدارها:', error);
            this.activeAlerts = [];
        }
    }

    renderSystemStatus() {
        // رندر وضعیت سیستم از داده واقعی API
        const container = document.querySelector('.status-grid');
        if (!container) return;

        const statusItems = [
            { 
                label: 'API CoinStats', 
                value: this.getAPIStatus(),
                status: this.getAPIStatus() === 'متصل' ? 'connected' : 'disconnected'
            },
            { 
                label: 'مدل AI', 
                value: this.getAIStatus(),
                status: this.getAIStatus() === 'فعال' ? 'active' : 'disconnected'
            },
            { 
                label: 'WebSocket', 
                value: this.getWebSocketStatus(),
                status: this.getWebSocketStatus() === 'متصل' ? 'connected' : 'disconnected'
            },
            { 
                label: 'دقت پیش‌بینی', 
                value: this.getAccuracy(),
                status: 'normal'
            }
        ];

        container.innerHTML = statusItems.map(item => `
            <div class="status-item">
                <div class="status-label">${item.label}</div>
                <div class="status-value ${item.status}">${item.value}</div>
            </div>
        `).join('');
    }

    getAPIStatus() {
        // بررسی وضعیت API از داده واقعی سیستم
        if (this.systemStatus.api_health && this.systemStatus.api_health.overall_status === 'healthy') {
            return 'متصل';
        }
        return 'قطع';
    }

    getAIStatus() {
        // بررسی وضعیت AI از داده واقعی سیستم
        if (this.systemStatus.ai_health && this.systemStatus.ai_health.overall_status === 'healthy') {
            return 'فعال';
        }
        return 'غیرفعال';
    }

    getWebSocketStatus() {
        // بررسی وضعیت WebSocket از داده واقعی سیستم
        if (this.systemStatus.websocket_status === 'connected') {
            return 'متصل';
        }
        return 'قطع';
    }

    getAccuracy() {
        // دقت از داده واقعی سیستم
        if (this.systemStatus.ai_health && this.systemStatus.ai_health.accuracy) {
            return `${Math.round(this.systemStatus.ai_health.accuracy.avg_confidence * 100)}%`;
        }
        return 'درحال محاسبه';
    }

    renderSystemStatusError() {
        const container = document.querySelector('.status-grid');
        if (!container) return;

        container.innerHTML = `
            <div class="status-item full-width">
                <div class="status-label">وضعیت سیستم</div>
                <div class="status-value error">خطا در اتصال به API</div>
            </div>
        `;
    }

    renderMarketData() {
        // رندر داده‌های واقعی بازار از API اسکن
        this.renderPriceDisplay();
        this.renderActiveSignals();
    }

    renderPriceDisplay() {
        const priceElement = document.querySelector('.quick-chart .current-price');
        const changeElement = document.querySelector('.quick-chart .price-change');
        
        if (this.marketData.length > 0) {
            const btcData = this.marketData.find(item => item.symbol === 'BTC');
            if (btcData) {
                priceElement.textContent = `$${btcData.current_price.toLocaleString('en-US', {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2
                })}`;
                
                const change = btcData.change || 0;
                changeElement.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`;
                changeElement.className = `price-change ${change >= 0 ? 'positive' : 'negative'}`;
                return;
            }
        }
        
        // اگر داده BTC نداریم
        priceElement.textContent = '---';
        changeElement.textContent = 'داده موجود نیست';
        changeElement.className = 'price-change error';
    }

    renderActiveSignals() {
        const container = document.getElementById('signalsList');
        if (!container) return;

        if (this.marketData.length === 0) {
            container.innerHTML = '<div class="no-data">داده‌ای برای نمایش موجود نیست</div>';
            return;
        }

        // فیلتر سیگنال‌های با اعتماد بالا
        const strongSignals = this.marketData.filter(item => 
            item.ai_signal && item.ai_signal.confidence > 0.7
        );

        if (strongSignals.length === 0) {
            container.innerHTML = '<div class="no-data">سیگنال قوی یافت نشد</div>';
            return;
        }

        container.innerHTML = strongSignals.map(signal => `
            <div class="signal-item ${signal.ai_signal.primary_signal.toLowerCase()}" 
                 onclick="window.location.href='/analysis?symbol=${signal.symbol}'">
                <div class="signal-info">
                    <div class="signal-symbol">${signal.symbol}</div>
                    <div class="signal-name">${this.getCoinName(signal.symbol)}</div>
                </div>
                <div class="signal-price">$${signal.current_price.toLocaleString()}</div>
                <div class="signal-change ${signal.change >= 0 ? 'positive' : 'negative'}">
                    ${signal.change >= 0 ? '+' : ''}${signal.change.toFixed(2)}%
                </div>
                <div class="signal-confidence">${Math.round(signal.ai_signal.confidence * 100)}%</div>
            </div>
        `).join('');
    }

    renderMarketDataError() {
        const priceElement = document.querySelector('.quick-chart .current-price');
        const changeElement = document.querySelector('.quick-chart .price-change');
        const signalsContainer = document.getElementById('signalsList');
        
        if (priceElement) priceElement.textContent = '---';
        if (changeElement) {
            changeElement.textContent = 'خطا در دریافت داده';
            changeElement.className = 'price-change error';
        }
        if (signalsContainer) {
            signalsContainer.innerHTML = '<div class="no-data">خطا در اتصال به API بازار</div>';
        }
    }

    renderActiveAlerts() {
        const container = document.getElementById('alertsList');
        if (!container) return;

        if (this.activeAlerts.length === 0) {
            container.innerHTML = '<div class="no-data">هشدار فعالی وجود ندارد</div>';
            return;
        }

        // فقط هشدارهای critical نمایش داده بشن
        const criticalAlerts = this.activeAlerts.filter(alert => 
            alert.level === 'critical' || alert.level === 'high'
        ).slice(0, 3); // حداکثر ۳ هشدار

        container.innerHTML = criticalAlerts.map(alert => `
            <div class="alert-item critical" onclick="window.location.href='/health#alerts'">
                <div class="alert-icon">⚠️</div>
                <div class="alert-content">
                    <div class="alert-title">${alert.title}</div>
                    <div class="alert-desc">${alert.message}</div>
                </div>
            </div>
        `).join('');
    }

    getCoinName(symbol) {
        const names = {
            'BTC': 'Bitcoin', 'ETH': 'Ethereum', 'SOL': 'Solana', 'ADA': 'Cardano',
            'DOT': 'Polkadot', 'LINK': 'Chainlink', 'BNB': 'Binance Coin', 
            'XRP': 'Ripple', 'DOGE': 'Dogecoin', 'MATIC': 'Polygon'
        };
        return names[symbol] || symbol;
    }

    setupEventListeners() {
        document.getElementById('alertsList')?.addEventListener('click', () => {
            window.location.href = '/health#alerts';
        });

        document.getElementById('signalsList')?.addEventListener('click', () => {
            window.location.href = '/analysis';
        });

        document.querySelectorAll('.quick-card').forEach(card => {
            card.addEventListener('click', () => {
                const page = card.dataset.page;
                if (page) window.location.href = page;
            });
        });
    }

    setupChart() {
        // نمودار با داده واقعی
        this.loadRealChartData();
    }

    async loadRealChartData() {
        try {
            const response = await fetch('/api/ai/analysis?symbols=BTC&period=24h');
            if (!response.ok) throw new Error('Chart API error');
            
            const data = await response.json();
            if (data.status === 'success') {
                this.renderRealChart(data.analysis_report);
            } else {
                this.showChartError('داده نمودار در دسترس نیست');
            }
        } catch (error) {
            console.error('Error loading chart data:', error);
            this.showChartError('خطا در دریافت داده نمودار');
        }
    }

    renderRealChart(analysisReport) {
        const container = document.getElementById('btcChart');
        if (!container) return;

        const btcData = analysisReport.symbol_analysis?.BTC;
        if (!btcData) {
            this.showChartError('داده BTC یافت نشد');
            return;
        }

        const prices = this.extractPricesFromData(btcData);
        if (prices.length === 0) {
            this.showChartError('داده قیمتی موجود نیست');
            return;
        }

        this.renderChart(container, prices);
    }

    extractPricesFromData(btcData) {
        try {
            if (btcData.historical_data?.result) {
                return btcData.historical_data.result
                    .slice(-20)
                    .map(item => {
                        const price = item.price || item.close || item.last;
                        return price && !isNaN(price) ? parseFloat(price) : null;
                    })
                    .filter(price => price !== null);
            }
        } catch (error) {
            console.error('Error extracting prices:', error);
        }
        return [];
    }

    renderChart(container, prices) {
        if (prices.length === 0) return;

        const maxPrice = Math.max(...prices);
        const minPrice = Math.min(...prices);
        const range = maxPrice - minPrice || 1;

        container.innerHTML = '';
        const chart = document.createElement('div');
        chart.className = 'simple-chart';
        chart.style.cssText = `
            width: 100%; height: 100%; display: flex; align-items: flex-end; 
            gap: 2px; padding: 10px;
        `;

        prices.forEach((price, index) => {
            const bar = document.createElement('div');
            const height = ((price - minPrice) / range) * 80;
            const isGreen = index === 0 || price >= prices[index - 1];
            
            bar.style.cssText = `
                flex: 1; height: ${height}%;
                background: ${isGreen ? 'var(--accent-success)' : 'var(--accent-danger)'};
                border-radius: 2px; opacity: ${0.6 + (index * 0.02)};
                transition: all 0.3s ease;
            `;
            
            bar.title = `$${price.toFixed(2)}`;
            chart.appendChild(bar);
        });

        container.appendChild(chart);
    }

    showChartError(message) {
        const container = document.getElementById('btcChart');
        if (container) {
            container.innerHTML = `<div class="chart-error">${message}</div>`;
        }
    }

    startRealTimeUpdates() {
        // بروزرسانی هر 30 ثانیه
        setInterval(async () => {
            await this.loadSystemStatus();
            await this.loadMarketData();
            await this.loadActiveAlerts();
        }, 30000);
    }
}

// راه‌اندازی
document.addEventListener('DOMContentLoaded', () => {
    new Dashboard();
});
