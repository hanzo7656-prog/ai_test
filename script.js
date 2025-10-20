// تنظیمات اصلی
const CONFIG = {
    API_BASE_URL: 'https://server-test-ovta.onrender.com/api',
    REFRESH_INTERVAL: 10000, // 10 ثانیه
    CACHE_DURATION: 30000 // 30 ثانیه
};

// وضعیت برنامه
const AppState = {
    isConnected: false,
    lastUpdate: null,
    currentData: null
};

// راه‌اندازی برنامه
class VortexApp {
    constructor() {
        this.init();
    }

    async init() {
        console.log('🚀 راه‌اندازی Vortex AI...');
        
        // راه‌اندازی ماژول‌ها
        this.setupEventListeners();
        this.startDataStream();
        
        // تست اولیه اتصال
        await this.testConnection();
        
        console.log('✅ Vortex AI آماده است');
    }

    setupEventListeners() {
        // فیلترهای زمانی
        document.querySelectorAll('.time-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.time-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                this.loadChartData(e.target.dataset.time);
            });
        });

        // رفرش دستی
        document.addEventListener('keydown', (e) => {
            if (e.key === 'r' && e.ctrlKey) {
                e.preventDefault();
                this.forceRefresh();
            }
        });
    }

    async testConnection() {
        try {
            const response = await fetch(`${CONFIG.API_BASE_URL}/health`);
            AppState.isConnected = response.ok;
            
            if (AppState.isConnected) {
                this.showNotification('اتصال به سرور برقرار شد', 'success');
            } else {
                this.showNotification('خطا در اتصال به سرور', 'error');
            }
        } catch (error) {
            AppState.isConnected = false;
            this.showNotification('سرور در دسترس نیست', 'error');
        }
    }

    async fetchMarketData() {
        if (!AppState.isConnected) return;

        try {
            const endpoints = [
                '/insights/dashboard',
                '/insights/fear-greed', 
                '/insights/btc-dominance',
                '/markets/cap'
            ];

            const promises = endpoints.map(endpoint => 
                fetch(`${CONFIG.API_BASE_URL}${endpoint}`)
                    .then(r => r.ok ? r.json() : null)
                    .catch(() => null)
            );

            const results = await Promise.all(promises);
            this.processData(results);
            
        } catch (error) {
            console.error('خطا در دریافت داده:', error);
            this.showNotification('خطا در دریافت داده‌ها', 'error');
        }
    }

    processData(data) {
        AppState.currentData = data;
        AppState.lastUpdate = new Date();
        
        this.updateUI();
        this.updateLastUpdateTime();
    }

    updateUI() {
        if (!AppState.currentData) return;

        // به روزرسانی کارت‌های KPI
        this.updateKPICards();
        
        // به روزرسانی پیش‌بینی
        this.updatePrediction();
        
        // به روزرسانی تیکر
        this.updateTicker();
    }

    updateKPICards() {
        // اینجا داده‌های واقعی رو با داده‌های نمونه جایگزین کن
        const cards = {
            'fear-greed': { value: 54, change: 2.1 },
            'btc-dominance': { value: '48.5%', change: -1.2 },
            'market-cap': { value: '2.45T', change: 3.8 },
            'ai-confidence': { value: '78%' }
        };

        Object.entries(cards).forEach(([type, data]) => {
            const card = document.querySelector(`[data-type="${type}"]`);
            if (card) {
                const valueEl = card.querySelector('.kpi-value');
                const changeEl = card.querySelector('.kpi-change');
                
                if (valueEl) valueEl.textContent = data.value;
                if (changeEl && data.change) {
                    changeEl.textContent = `${data.change > 0 ? '+' : ''}${data.change}%`;
                    changeEl.className = `kpi-change ${data.change > 0 ? 'positive' : 'negative'}`;
                }
            }
        });
    }

    updatePrediction() {
        // به روزرسانی پیش‌بینی بر اساس داده‌های واقعی
        const predictionCard = document.querySelector('.prediction-card');
        if (predictionCard) {
            // اینجا منطق پیش‌بینی واقعی رو پیاده‌سازی کن
            const confidence = 78;
            const trend = 'bullish';
            
            const badge = predictionCard.querySelector('.prediction-badge');
            const confidenceFill = predictionCard.querySelector('.confidence-fill');
            const levelText = predictionCard.querySelector('.level-text');
            
            if (badge) {
                badge.textContent = trend === 'bullish' ? 'صعودی' : 'نزولی';
                badge.className = `prediction-badge ${trend}`;
            }
            
            if (confidenceFill) {
                confidenceFill.style.width = `${confidence}%`;
            }
            
            if (levelText) {
                levelText.textContent = `سطح اطمینان: ${confidence}%`;
            }
        }
    }

    updateTicker() {
        // به روزرسانی داده‌های زنده
        const tickerData = [
            { name: 'BTC', price: '$51,234', change: 2.3 },
            { name: 'ETH', price: '$2,856', change: 1.8 },
            { name: 'SOL', price: '$102.45', change: -0.5 }
        ];

        const tickerContent = document.querySelector('.ticker-content');
        if (tickerContent) {
            tickerContent.innerHTML = tickerData.map(coin => `
                <div class="coin-ticker">
                    <span class="coin-name">${coin.name}</span>
                    <span class="coin-price">${coin.price}</span>
                    <span class="coin-change ${coin.change > 0 ? 'positive' : 'negative'}">
                        ${coin.change > 0 ? '+' : ''}${coin.change}%
                    </span>
                </div>
            `).join('');
        }
    }

    updateLastUpdateTime() {
        const updateEl = document.getElementById('last-update');
        if (updateEl) {
            const now = new Date();
            updateEl.textContent = now.toLocaleTimeString('fa-IR');
        }
    }

    loadChartData(timeframe) {
        // اینجا نمودار واقعی رو لود کن
        console.log(`لود نمودار برای ${timeframe}`);
        
        const placeholder = document.querySelector('.chart-placeholder');
        if (placeholder) {
            placeholder.innerHTML = `
                <i class="fas fa-chart-line"></i>
                <p>نمودار ${timeframe} درحال بارگذاری...</p>
            `;
        }
    }

    startDataStream() {
        // شروع دریافت داده‌های زنده
        setInterval(() => {
            this.fetchMarketData();
        }, CONFIG.REFRESH_INTERVAL);

        // اولین درخواست
        this.fetchMarketData();
    }

    forceRefresh() {
        this.showNotification('بروزرسانی دستی داده‌ها...', 'info');
        this.fetchMarketData();
    }

    showNotification(message, type = 'info') {
        // نمایش نوتیفیکیشن ساده
        console.log(`[${type.toUpperCase()}] ${message}`);
        
        // می‌تونی اینجا یه سیستم نوتیفیکیشن زیبا اضافه کنی
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            left: 20px;
            background: ${type === 'error' ? '#ff2a6d' : type === 'success' ? '#00ff88' : '#00d4ff'};
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            z-index: 1000;
            animation: slideIn 0.3s ease;
        `;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }
}

// راه‌اندازی برنامه وقتی DOM لود شد
document.addEventListener('DOMContentLoaded', () => {
    new VortexApp();
});

// اضافه کردن انیمیشن slideIn
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(-100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
`;
document.head.appendChild(style);
