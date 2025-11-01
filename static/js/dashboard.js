// static/js/dashboard.js - استفاده از endpoint های درست
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
            console.log('🔄 دریافت وضعیت سیستم...');
            
            // استفاده از endpoint درست
            const response = await fetch('/api/system/status');
            
            if (!response.ok) {
                throw new Error(`خطای API: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('📊 وضعیت سیستم:', data);
            
            if (data.status === 'success') {
                this.systemStatus = data;
                this.renderSystemStatus();
            } else {
                throw new Error('داده معتبر دریافت نشد');
            }
            
        } catch (error) {
            console.error('❌ خطا در دریافت وضعیت سیستم:', error);
            this.renderSystemStatusError();
        }
    }

    async loadMarketData() {
        try {
            console.log('🔄 دریافت داده‌های بازار...');
            
            // استفاده از endpoint اسکن سریع
            const response = await fetch('/api/ai/scan', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            if (!response.ok) {
                throw new Error(`خطای اسکن: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('📊 داده‌های بازار:', data);

            if (data.status === 'success' && data.scan_results) {
                this.marketData = data.scan_results;
                this.renderMarketData();
            } else {
                throw new Error('داده معتبر از اسکن دریافت نشد');
            }

        } catch (error) {
            console.error('❌ خطا در دریافت داده بازار:', error);
            this.renderMarketDataError();
        }
    }

    async loadActiveAlerts() {
        try {
            console.log('🔄 دریافت هشدارها...');
            
            // استفاده از endpoint سلامت سیستم
            const response = await fetch('/api/system/health');
            if (response.ok) {
                const data = await response.json();
                this.activeAlerts = data.system_health?.active_alerts || [];
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

        container.style.cursor = 'pointer';

        const statusItems = [
            { 
                label: 'API CoinStats', 
                value: this.systemStatus.api_health?.coinstats === 'connected' ? 'متصل' : 'قطع',
                status: this.systemStatus.api_health?.coinstats === 'connected' ? 'connected' : 'disconnected'
            },
            { 
                label: 'مدل AI', 
                value: this.systemStatus.ai_health?.status === 'active' ? 'فعال' : 'غیرفعال',
                status: this.systemStatus.ai_health?.status === 'active' ? 'active' : 'disconnected'
            },
            { 
                label: 'WebSocket', 
                value: this.systemStatus.api_health?.websocket === 'connected' ? 'متصل' : 'قطع',
                status: this.systemStatus.api_health?.websocket === 'connected' ? 'connected' : 'disconnected'
            },
            { 
                label: 'دقت پیش‌بینی', 
                value: this.systemStatus.ai_health?.accuracy ? `${Math.round(this.systemStatus.ai_health.accuracy * 100)}%` : 'درحال محاسبه',
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

    renderMarketData() {
        this.renderPriceDisplay();
        this.renderActiveSignals();
    }

    renderPriceDisplay() {
        const priceElement = document.querySelector('.quick-chart .current-price');
        const changeElement = document.querySelector('.quick-chart .price-change');
        const chartContainer = document.querySelector('.quick-chart');
        
        if (chartContainer) {
            chartContainer.style.cursor = 'pointer';
        }
        
        if (this.marketData.length > 0) {
            const btcData = this.marketData.find(item => item.symbol === 'BTC');
            if (btcData) {
                priceElement.textContent = `$${btcData.current_price.toLocaleString()}`;
                
                const change = btcData.change || 0;
                changeElement.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`;
                changeElement.className = `price-change ${change >= 0 ? 'positive' : 'negative'}`;
                return;
            }
        }
        
        priceElement.textContent = '---';
        changeElement.textContent = 'داده موجود نیست';
        changeElement.className = 'price-change error';
    }

    renderActiveSignals() {
        const container = document.getElementById('signalsList');
        if (!container) return;

        container.style.cursor = 'pointer';

        if (this.marketData.length === 0) {
            container.innerHTML = '<div class="no-data">داده‌ای برای نمایش موجود نیست</div>';
            return;
        }

        // فیلتر سیگنال‌های قوی
        const strongSignals = this.marketData.filter(item => 
            item.ai_signal && item.ai_signal.confidence > 0.6
        ).slice(0, 4); // حداکثر ۴ سیگنال

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
                <div class="signal-price">$${signal.current_price.toLocaleString()}</div>
                <div class="signal-change ${signal.change >= 0 ? 'positive' : 'negative'}">
                    ${signal.change >= 0 ? '+' : ''}${signal.change.toFixed(2)}%
                </div>
                <div class="signal-confidence">${Math.round(signal.ai_signal.confidence * 100)}%</div>
            </div>
        `).join('');
    }

    // بقیه متدها مانند قبل...

    setupEventListeners() {
        console.log('🎯 راه‌اندازی event listener ها...');

        // کلیک روی هشدارها
        const alertsList = document.getElementById('alertsList');
        if (alertsList) {
            alertsList.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                window.location.href = '/health#alerts';
            });
        }

        // کلیک روی سیگنال‌ها
        const signalsList = document.getElementById('signalsList');
        if (signalsList) {
            signalsList.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                window.location.href = '/analysis';
            });
        }

        // کلیک روی وضعیت سیستم
        const systemStatus = document.querySelector('.system-status');
        if (systemStatus) {
            systemStatus.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                window.location.href = '/health';
            });
        }

        // کلیک روی نمودار
        const quickChart = document.querySelector('.quick-chart');
        if (quickChart) {
            quickChart.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                window.location.href = '/analysis';
            });
        }

        // کلیک روی کارت‌های سریع
        document.querySelectorAll('.quick-card').forEach((card) => {
            card.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                const page = card.dataset.page;
                if (page) {
                    window.location.href = page;
                }
            });
        });

        console.log('✅ event listener ها راه‌اندازی شدند');
    }

    getCoinName(symbol) {
        const names = {
            'BTC': 'Bitcoin', 'ETH': 'Ethereum', 'SOL': 'Solana', 'ADA': 'Cardano',
            'DOT': 'Polkadot', 'LINK': 'Chainlink', 'BNB': 'Binance Coin', 
            'XRP': 'Ripple', 'DOGE': 'Dogecoin', 'MATIC': 'Polygon'
        };
        return names[symbol] || symbol;
    }

    startRealTimeUpdates() {
        setInterval(async () => {
            await this.loadSystemStatus();
            await this.loadMarketData();
            await this.loadActiveAlerts();
        }, 30000);
    }
}

// راه‌اندازی
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 راه‌اندازی داشبورد...');
    setTimeout(() => {
        try {
            new Dashboard();
            console.log('✅ داشبورد با موفقیت راه‌اندازی شد');
        } catch (error) {
            console.error('❌ خطا در راه‌اندازی داشبورد:', error);
        }
    }, 1000);
});
