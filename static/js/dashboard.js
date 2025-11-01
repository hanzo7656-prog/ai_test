// static/js/dashboard.js - اصلاح event listener ها
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
        this.setupEventListeners(); // این باید اول صدا زده بشه
        this.startRealTimeUpdates();
    }

    setupEventListeners() {
        console.log('🎯 راه‌اندازی event listener های داشبورد...');
        
        // کلیک روی هشدارها - رفتن به صفحه سلامت
        const alertsList = document.getElementById('alertsList');
        if (alertsList) {
            alertsList.addEventListener('click', (e) => {
                console.log('⚠️ کلیک روی هشدارها');
                e.preventDefault();
                e.stopPropagation();
                window.location.href = '/health#alerts';
            });
        }

        // کلیک روی سیگنال‌ها - رفتن به صفحه تحلیل
        const signalsList = document.getElementById('signalsList');
        if (signalsList) {
            signalsList.addEventListener('click', (e) => {
                console.log('📈 کلیک روی سیگنال‌ها');
                e.preventDefault();
                e.stopPropagation();
                window.location.href = '/analysis';
            });
        }

        // کلیک روی کارت‌های سریع دسترسی
        document.querySelectorAll('.quick-card').forEach((card, index) => {
            card.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                
                const page = card.dataset.page;
                console.log(`🚀 کلیک روی کارت ${index + 1}: ${page}`);
                
                if (page) {
                    window.location.href = page;
                }
            });
        });

        // کلیک روی وضعیت سیستم - رفتن به صفحه سلامت
        const systemStatus = document.querySelector('.system-status');
        if (systemStatus) {
            systemStatus.addEventListener('click', (e) => {
                console.log('🖥️ کلیک روی وضعیت سیستم');
                e.preventDefault();
                e.stopPropagation();
                window.location.href = '/health';
            });
        }

        // کلیک روی نمودار - رفتن به صفحه تحلیل
        const quickChart = document.querySelector('.quick-chart');
        if (quickChart) {
            quickChart.addEventListener('click', (e) => {
                console.log('📊 کلیک روی نمودار');
                e.preventDefault();
                e.stopPropagation();
                window.location.href = '/analysis';
            });
        }

        console.log('✅ همه event listener ها راه‌اندازی شدند');
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
            
            // استفاده از API تحلیل برای داده‌های سریع
            const response = await fetch('/api/ai/analysis?symbols=BTC,ETH,SOL,ADA&period=1h');
            
            if (!response.ok) {
                throw new Error(`API تحلیل خطا: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('📊 داده‌های بازار:', data);

            if (data.status === 'success' && data.analysis_report) {
                this.processMarketData(data.analysis_report);
            } else {
                throw new Error('داده معتبر از API تحلیل دریافت نشد');
            }

        } catch (error) {
            console.error('❌ خطا در دریافت داده بازار:', error);
            this.renderMarketDataError();
        }
    }

    processMarketData(analysisReport) {
        if (!analysisReport.symbol_analysis) {
            this.renderMarketDataError();
            return;
        }

        this.marketData = analysisReport.symbol_analysis;
        this.renderMarketData();
    }

    async loadActiveAlerts() {
        try {
            console.log('🔄 دریافت هشدارهای فعال...');
            
            // استفاده از API سلامت برای هشدارها
            const response = await fetch('/api/system/health');
            if (response.ok) {
                const data = await response.json();
                this.activeAlerts = data.active_alerts || [];
                this.renderActiveAlerts();
            }
        } catch (error) {
            console.error('❌ خطا در دریافت هشدارها:', error);
            this.activeAlerts = [];
        }
    }

    renderSystemStatus() {
        const container = document.querySelector('.status-grid');
        if (!container) return;

        // اضافه کردن cursor pointer برای قابلیت کلیک
        container.style.cursor = 'pointer';

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
        if (this.systemStatus.api_health && this.systemStatus.api_health.overall_status === 'healthy') {
            return 'متصل';
        }
        return 'قطع';
    }

    getAIStatus() {
        if (this.systemStatus.ai_health && this.systemStatus.ai_health.overall_status === 'healthy') {
            return 'فعال';
        }
        return 'غیرفعال';
    }

    getWebSocketStatus() {
        if (this.systemStatus.websocket_status === 'connected') {
            return 'متصل';
        }
        return 'قطع';
    }

    getAccuracy() {
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
        this.renderPriceDisplay();
        this.renderActiveSignals();
    }

    renderPriceDisplay() {
        const priceElement = document.querySelector('.quick-chart .current-price');
        const changeElement = document.querySelector('.quick-chart .price-change');
        const chartContainer = document.querySelector('.quick-chart');
        
        // اضافه کردن cursor pointer برای نمودار
        if (chartContainer) {
            chartContainer.style.cursor = 'pointer';
        }
        
        if (this.marketData.BTC) {
            const btcData = this.marketData.BTC;
            const price = btcData.current_price || 0;
            const change = btcData.technical_score ? (btcData.technical_score - 0.5) * 10 : 0;
            
            priceElement.textContent = `$${price.toLocaleString('en-US', {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2
            })}`;
            
            changeElement.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`;
            changeElement.className = `price-change ${change >= 0 ? 'positive' : 'negative'}`;
        } else {
            priceElement.textContent = '---';
            changeElement.textContent = 'داده موجود نیست';
            changeElement.className = 'price-change error';
        }
    }

    renderActiveSignals() {
        const container = document.getElementById('signalsList');
        if (!container) return;

        // اضافه کردن cursor pointer برای لیست سیگنال‌ها
        container.style.cursor = 'pointer';

        const signals = [];
        
        Object.entries(this.marketData).forEach(([symbol, data]) => {
            if (data.ai_signal && data.ai_signal.signals) {
                const signal = data.ai_signal.signals;
                signals.push({
                    symbol: symbol,
                    name: this.getCoinName(symbol),
                    price: data.current_price || 0,
                    change: (data.technical_score - 0.5) * 10 || 0,
                    type: signal.primary_signal.toLowerCase(),
                    confidence: Math.round(signal.signal_confidence * 100)
                });
            }
        });

        if (signals.length === 0) {
            container.innerHTML = '<div class="no-data">سیگنال فعالی یافت نشد</div>';
            return;
        }

        container.innerHTML = signals.map(signal => `
            <div class="signal-item ${signal.type}">
                <div class="signal-info">
                    <div class="signal-symbol">${signal.symbol}</div>
                    <div class="signal-name">${signal.name}</div>
                </div>
                <div class="signal-price">$${signal.price.toLocaleString()}</div>
                <div class="signal-change ${signal.change >= 0 ? 'positive' : 'negative'}">
                    ${signal.change >= 0 ? '+' : ''}${signal.change.toFixed(2)}%
                </div>
                <div class="signal-confidence">${signal.confidence}%</div>
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

        // اضافه کردن cursor pointer برای هشدارها
        container.style.cursor = 'pointer';

        if (this.activeAlerts.length === 0) {
            container.innerHTML = '<div class="no-data">هشدار فعالی وجود ندارد</div>';
            return;
        }

        container.innerHTML = this.activeAlerts.slice(0, 3).map(alert => `
            <div class="alert-item critical">
                <div class="alert-icon">⚠️</div>
                <div class="alert-content">
                    <div class="alert-title">${alert.title || 'هشدار سیستم'}</div>
                    <div class="alert-desc">${alert.message || 'مشکل در سیستم شناسایی شد'}</div>
                </div>
            </div>
        `).join('');
    }

    getCoinName(symbol) {
        const names = {
            'BTC': 'Bitcoin', 'ETH': 'Ethereum', 'SOL': 'Solana', 'ADA': 'Cardano'
        };
        return names[symbol] || symbol;
    }

    setupChart() {
        // نمودار ساده
        this.renderSampleChart();
    }

    renderSampleChart() {
        const container = document.getElementById('btcChart');
        if (!container) return;

        // اضافه کردن cursor pointer برای نمودار
        container.style.cursor = 'pointer';

        const prices = Array.from({length: 20}, (_, i) => {
            return 43000 + Math.sin(i * 0.5) * 500 + Math.random() * 300;
        });

        const maxPrice = Math.max(...prices);
        const minPrice = Math.min(...prices);
        const range = maxPrice - minPrice || 1;

        container.innerHTML = '';
        const chart = document.createElement('div');
        chart.className = 'simple-chart';
        chart.style.cssText = `
            width: 100%; height: 100%; display: flex; align-items: flex-end; 
            gap: 2px; padding: 10px; cursor: pointer;
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

    startRealTimeUpdates() {
        setInterval(async () => {
            await this.loadSystemStatus();
            await this.loadMarketData();
            await this.loadActiveAlerts();
        }, 30000);
    }
}

// راه‌اندازی با تاخیر برای اطمینان از لود کامل DOM
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 DOM Ready - Starting Dashboard...');
    
    // تاخیر برای اطمینان از لود کامل المان‌ها
    setTimeout(() => {
        try {
            new Dashboard();
            console.log('✅ Dashboard Successfully Initialized');
        } catch (error) {
            console.error('❌ Dashboard Initialization Error:', error);
        }
    }, 500);
});
