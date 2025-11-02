// static/js/dashboard.js - کاملاً اصلاح شده و یکپارچه
class Dashboard {
    constructor() {
        this.systemStatus = {};
        this.marketData = [];
        this.activeAlerts = [];
        this.systemMetrics = {};
        this.updateInterval = null;
        this.isInitialized = false;
        
        this.initializeDashboard();
    }

    async initializeDashboard() {
        if (this.isInitialized) return;
        
        console.log('🚀 راه‌اندازی داشبورد...');
        
        try {
            // لود همزمان با مدیریت خطا
            const results = await Promise.allSettled([
                this.loadSystemStatus(),
                this.loadMarketData(),
                this.loadActiveAlerts(),
                this.loadSystemMetrics()
            ]);

            // بررسی نتایج
            results.forEach((result, index) => {
                if (result.status === 'rejected') {
                    console.error(`خطا در کامپوننت ${index}:`, result.reason);
                }
            });

            this.setupEventListeners();
            this.setupChart();
            this.startRealTimeUpdates();
            
            this.isInitialized = true;
            console.log('✅ داشبورد با موفقیت راه‌اندازی شد');
            
        } catch (error) {
            console.error('❌ خطا در راه‌اندازی داشبورد:', error);
            this.showGlobalError('خطا در راه‌اندازی داشبورد');
        }
    }

    async loadSystemStatus() {
        try {
            console.log('🔄 دریافت وضعیت سیستم...');
            const response = await fetch('/api/system/status');
            
            if (!response.ok) throw new Error(`خطای API: ${response.status}`);
            
            const data = await response.json();
            console.log('📊 وضعیت سیستم:', data);
            
            if (data.status === 'success') {
                this.systemStatus = data;
                this.renderSystemStatus();
                
                // به روزرسانی state全局
                window.appState = window.appState || {};
                window.appState.systemStatus = data;
                
            } else {
                throw new Error('داده معتبر دریافت نشد');
            }
            
        } catch (error) {
            console.error('❌ خطا در دریافت وضعیت سیستم:', error);
            this.renderSystemStatusError('خطا در دریافت وضعیت سیستم');
        }
    }

    async loadMarketData() {
        try {
            console.log('🔄 دریافت داده‌های بازار...');
            const response = await fetch('/api/ai/scan', { 
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) throw new Error(`خطای اسکن: ${response.status}`);
            
            const data = await response.json();
            console.log('📊 داده‌های بازار:', data);

            if (data.status === 'success' && data.scan_results) {
                this.marketData = data.scan_results;
                this.renderMarketData();
                
                // به روزرسانی state全局
                window.appState = window.appState || {};
                window.appState.marketData = data.scan_results;
                window.appState.lastScanTime = new Date().toISOString();
                
            } else {
                throw new Error('داده معتبر از اسکن دریافت نشد');
            }

        } catch (error) {
            console.error('❌ خطا در دریافت داده بازار:', error);
            this.renderMarketDataError('خطا در دریافت داده‌های بازار');
        }
    }

    async loadActiveAlerts() {
        try {
            console.log('🔄 دریافت هشدارها...');
            const response = await fetch('/api/system/alerts');
            
            if (response.ok) {
                const data = await response.json();
                this.activeAlerts = data.alerts || [];
                this.renderActiveAlerts();
                
                // به روزرسانی state全局
                window.appState = window.appState || {};
                window.appState.activeAlerts = data.alerts || [];
                
            } else {
                throw new Error(`خطای API: ${response.status}`);
            }
        } catch (error) {
            console.error('❌ خطا در دریافت هشدارها:', error);
            this.activeAlerts = [];
        }
    }

    async loadSystemMetrics() {
        try {
            console.log('🔄 دریافت متریک‌های سیستم...');
            const response = await fetch('/api/system/metrics');
            
            if (response.ok) {
                const data = await response.json();
                this.systemMetrics = data.current_metrics || {};
                this.renderSystemMetrics();
                
                // به روزرسانی state全局
                window.appState = window.appState || {};
                window.appState.systemMetrics = data.current_metrics || {};
                
            } else {
                throw new Error(`خطای API: ${response.status}`);
            }
        } catch (error) {
            console.error('❌ خطا در دریافت متریک‌ها:', error);
            this.systemMetrics = {};
        }
    }

    renderSystemStatus() {
        const container = document.querySelector('.status-grid');
        if (!container) {
            console.warn('❌ container وضعیت سیستم یافت نشد');
            return;
        }

        container.style.cursor = 'pointer';

        const statusItems = [
            { 
                label: 'API CoinStats', 
                value: this.systemStatus.api_health?.coinstats === 'connected' ? 'متصل' : 'قطع',
                status: this.systemStatus.api_health?.coinstats === 'connected' ? 'connected' : 'disconnected',
                data: this.systemStatus.api_health?.coinstats
            },
            { 
                label: 'مدل AI', 
                value: this.systemStatus.ai_health?.status === 'active' ? 'فعال' : 'غیرفعال',
                status: this.systemStatus.ai_health?.status === 'active' ? 'active' : 'disconnected',
                data: this.systemStatus.ai_health?.status
            },
            { 
                label: 'WebSocket', 
                value: this.systemStatus.api_health?.websocket === 'connected' ? 'متصل' : 'قطع',
                status: this.systemStatus.api_health?.websocket === 'connected' ? 'connected' : 'disconnected',
                data: this.systemStatus.api_health?.websocket
            },
            { 
                label: 'دقت پیش‌بینی', 
                value: this.systemStatus.ai_health?.accuracy ? `${Math.round(this.systemStatus.ai_health.accuracy * 100)}%` : 'درحال محاسبه',
                status: 'normal',
                data: this.systemStatus.ai_health?.accuracy
            }
        ];

        container.innerHTML = statusItems.map(item => `
            <div class="status-item" data-status="${item.data}">
                <div class="status-label">${item.label}</div>
                <div class="status-value ${item.status}">${item.value}</div>
            </div>
        `).join('');
    }

    renderMarketData() {
        this.renderPriceDisplay();
        this.renderActiveSignals();
    }

    renderPriceDisplay() {
        const priceElement = document.querySelector('.quick-chart .current-price');
        const changeElement = document.querySelector('.quick-chart .price-change');
        const chartContainer = document.querySelector('.quick-chart');
        
        if (!priceElement || !changeElement) {
            console.warn('❌ المنت‌های قیمت یافت نشدند');
            return;
        }
        
        if (chartContainer) chartContainer.style.cursor = 'pointer';
        
        if (this.marketData && this.marketData.length > 0) {
            const btcData = this.marketData.find(item => item.symbol === 'BTC');
            if (btcData && btcData.current_price) {
                priceElement.textContent = `$${btcData.current_price.toLocaleString()}`;
                
                const change = btcData.change || 0;
                changeElement.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`;
                changeElement.className = `price-change ${change >= 0 ? 'positive' : 'negative'}`;
                
                // آپدیت عنوان
                const titleElement = document.querySelector('.quick-chart .section-header h2');
                if (titleElement) titleElement.textContent = `📊 ${btcData.symbol}/USDT`;
                return;
            }
        }
        
        // حالت خطا یا داده ناموجود
        priceElement.textContent = '---';
        changeElement.textContent = 'در حال دریافت...';
        changeElement.className = 'price-change loading';
    }

    renderActiveSignals() {
        const container = document.getElementById('signalsList');
        if (!container) {
            console.warn('❌ container سیگنال‌ها یافت نشد');
            return;
        }

        container.style.cursor = 'pointer';

        if (!this.marketData || this.marketData.length === 0) {
            container.innerHTML = '<div class="no-data">در حال دریافت داده‌های بازار...</div>';
            return;
        }

        // فیلتر سیگنال‌های قوی
        const strongSignals = this.marketData
            .filter(item => item.ai_signal && item.ai_signal.confidence > 0.6)
            .slice(0, 4);

        if (strongSignals.length === 0) {
            container.innerHTML = '<div class="no-data">سیگنال قوی یافت نشد</div>';
            return;
        }

        container.innerHTML = strongSignals.map(signal => `
            <div class="signal-item ${signal.ai_signal.primary_signal.toLowerCase()}">
                <div class="signal-info">
                    <div class="signal-symbol">${signal.symbol}</div>
                    <div class="signal-name">${this.getCoinName(signal.symbol)}</div>
                </div>
                <div class="signal-price">$${(signal.current_price || 0).toLocaleString()}</div>
                <div class="signal-change ${(signal.change || 0) >= 0 ? 'positive' : 'negative'}">
                    ${(signal.change || 0) >= 0 ? '+' : ''}${(signal.change || 0).toFixed(2)}%
                </div>
                <div class="signal-confidence">${Math.round((signal.ai_signal.confidence || 0) * 100)}%</div>
            </div>
        `).join('');
    }

    renderActiveAlerts() {
        const container = document.getElementById('alertsList');
        if (!container) {
            console.warn('❌ container هشدارها یافت نشد');
            return;
        }

        container.style.cursor = 'pointer';

        if (!this.activeAlerts || this.activeAlerts.length === 0) {
            container.innerHTML = '<div class="no-data">هشدار فعالی وجود ندارد</div>';
            return;
        }

        // فقط هشدارهای مهم
        const importantAlerts = this.activeAlerts
            .filter(alert => alert.level === 'critical' || alert.level === 'warning')
            .slice(0, 3);

        container.innerHTML = importantAlerts.map(alert => `
            <div class="alert-item ${alert.level}">
                <div class="alert-icon">${this.getAlertIcon(alert.level)}</div>
                <div class="alert-content">
                    <div class="alert-title">${alert.title || 'هشدار سیستم'}</div>
                    <div class="alert-desc">${alert.message || 'توضیحات موجود نیست'}</div>
                </div>
            </div>
        `).join('');
    }

    renderSystemMetrics() {
        console.log('📈 متریک‌های سیستم:', this.systemMetrics);
        // می‌توانید اینجا متریک‌ها را در UI نمایش دهید
    }

    renderSystemStatusError(message) {
        const container = document.querySelector('.status-grid');
        if (!container) return;

        container.innerHTML = `
            <div class="status-item full-width">
                <div class="status-label">وضعیت سیستم</div>
                <div class="status-value error">${message}</div>
            </div>
        `;
    }

    renderMarketDataError(message) {
        const priceElement = document.querySelector('.quick-chart .current-price');
        const changeElement = document.querySelector('.quick-chart .price-change');
        const signalsContainer = document.getElementById('signalsList');
        
        if (priceElement) priceElement.textContent = '---';
        if (changeElement) {
            changeElement.textContent = message;
            changeElement.className = 'price-change error';
        }
        if (signalsContainer) {
            signalsContainer.innerHTML = `<div class="no-data">${message}</div>`;
        }
    }

    getCoinName(symbol) {
        const names = {
            'BTC': 'Bitcoin', 'ETH': 'Ethereum', 'SOL': 'Solana', 'ADA': 'Cardano',
            'DOT': 'Polkadot', 'LINK': 'Chainlink', 'BNB': 'Binance Coin', 
            'XRP': 'Ripple', 'DOGE': 'Dogecoin', 'MATIC': 'Polygon',
            'LTC': 'Litecoin', 'BCH': 'Bitcoin Cash', 'XLM': 'Stellar',
            'ATOM': 'Cosmos', 'ETC': 'Ethereum Classic', 'XMR': 'Monero'
        };
        return names[symbol] || symbol;
    }

    getAlertIcon(level) {
        const icons = {
            'critical': '🚨',
            'warning': '⚠️', 
            'info': 'ℹ️'
        };
        return icons[level] || '⚠️';
    }

    setupEventListeners() {
        console.log('🎯 راه‌اندازی event listener ها...');

        // کلیک روی هشدارها
        this.setupClickListener('alertsList', '/health#alerts', 'هشدارها');

        // کلیک روی سیگنال‌ها
        this.setupClickListener('signalsList', '/analysis', 'سیگنال‌ها');

        // کلیک روی وضعیت سیستم
        this.setupClickListener('system-status', '/health', 'وضعیت سیستم');

        // کلیک روی نمودار
        this.setupClickListener('quick-chart', '/analysis', 'نمودار');

        // کلیک روی کارت‌های سریع
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

        console.log('✅ همه event listener ها راه‌اندازی شدند');
    }

    setupClickListener(elementId, targetUrl, description) {
        const element = document.getElementById(elementId) || document.querySelector(`.${elementId}`);
        if (element) {
            element.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                console.log(`🎯 کلیک روی ${description}`);
                window.location.href = targetUrl;
            });
        } else {
            console.warn(`❌ المنت ${elementId} برای کلیک یافت نشد`);
        }
    }

    setupChart() {
        this.renderSampleChart();
    }

    renderSampleChart() {
        const container = document.getElementById('btcChart');
        if (!container) {
            console.warn('❌ container نمودار یافت نشد');
            return;
        }

        container.style.cursor = 'pointer';

        // استفاده از داده واقعی اگر موجود باشد
        let prices;
        if (this.marketData && this.marketData.length > 0) {
            const btcData = this.marketData.find(item => item.symbol === 'BTC');
            if (btcData && btcData.historical_prices) {
                prices = btcData.historical_prices;
            }
        }

        // اگر داده واقعی نبود، از داده نمونه استفاده کن
        if (!prices) {
            prices = Array.from({length: 20}, (_, i) => {
                return 43000 + Math.sin(i * 0.5) * 500 + Math.random() * 300;
            });
        }

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
        console.log('🔄 شروع بروزرسانی‌های Real-time...');
        
        // پاک‌سازی interval قبلی اگر وجود دارد
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        
        // بروزرسانی هر 30 ثانیه
        this.updateInterval = setInterval(async () => {
            console.log('🔄 بروزرسانی Real-time داده‌ها...');
            await Promise.allSettled([
                this.loadSystemStatus(),
                this.loadMarketData(),
                this.loadActiveAlerts()
            ]);
        }, 30000);
    }

    showGlobalError(message) {
        // ایجاد نوتفیکیشن خطای سراسری
        const errorDiv = document.createElement('div');
        errorDiv.className = 'global-error';
        errorDiv.style.cssText = `
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: var(--accent-danger);
            color: white;
            padding: 1rem 2rem;
            border-radius: 8px;
            z-index: 10000;
            animation: slideDown 0.3s ease;
        `;
        errorDiv.textContent = message;
        
        document.body.appendChild(errorDiv);
        
        setTimeout(() => {
            errorDiv.remove();
        }, 5000);
    }

    // متد cleanup برای جلوگیری از memory leak
    destroy() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        this.isInitialized = false;
        console.log('🧹 داشبورد cleanup شد');
    }
}

// راه‌اندازی با تاخیر برای اطمینان از لود کامل DOM
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 DOM Ready - Starting Dashboard System...');
    
    // اطمینان از عدم راه‌اندازی تکراری
    if (window.dashboardInstance) {
        console.warn('⚠️ Dashboard instance already exists');
        return;
    }
    
    setTimeout(() => {
        try {
            window.dashboardInstance = new Dashboard();
            window.appState = window.appState || {};
            console.log('✅ Dashboard System Successfully Initialized');
        } catch (error) {
            console.error('❌ Dashboard System Initialization Error:', error);
        }
    }, 1000);
});

// مدیریت unload صفحه
window.addEventListener('beforeunload', function() {
    if (window.dashboardInstance) {
        window.dashboardInstance.destroy();
    }
});
