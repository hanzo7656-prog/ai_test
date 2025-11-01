// static/js/scan.js
class MarketScanner {
    constructor() {
        this.currentFilter = 'all';
        this.currentView = 'grid';
        this.scanResults = [];
        this.isScanning = false;
        
        this.initializeScanner();
        this.setupEventListeners();
        this.loadSampleData();
    }

    initializeScanner() {
        this.updateStats();
    }

    setupEventListeners() {
        // فیلترهای اسکن
        document.querySelectorAll('.filter-chip').forEach(chip => {
            chip.addEventListener('click', (e) => {
                this.setActiveFilter(e.currentTarget.dataset.filter);
            });
        });

        // اسلایدرهای تنظیمات
        document.getElementById('volumeFilter')?.addEventListener('input', (e) => {
            this.updateVolumeValue(e.target.value);
        });

        document.getElementById('changeFilter')?.addEventListener('input', (e) => {
            this.updateChangeValue(e.target.value);
        });

        // دکمه‌های اسکن
        document.getElementById('startScan')?.addEventListener('click', () => {
            this.startScan();
        });

        document.getElementById('advancedScan')?.addEventListener('click', () => {
            this.startAdvancedScan();
        });

        // مرتب‌سازی
        document.getElementById('sortBy')?.addEventListener('change', (e) => {
            this.sortResults(e.target.value);
        });

        // تغییر view
        document.querySelectorAll('.view-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.setActiveView(e.currentTarget.dataset.view);
            });
        });
    }

    setActiveFilter(filter) {
        this.currentFilter = filter;
        
        // آپدیت UI
        document.querySelectorAll('.filter-chip').forEach(chip => {
            chip.classList.remove('active');
        });
        document.querySelector(`[data-filter="${filter}"]`)?.classList.add('active');
        
        // فیلتر کردن نتایج
        this.filterResults();
    }

    updateVolumeValue(value) {
        const volumeValue = document.getElementById('volumeValue');
        if (volumeValue) {
            volumeValue.textContent = `${value}M`;
        }
    }

    updateChangeValue(value) {
        const changeValue = document.getElementById('changeValue');
        if (changeValue) {
            changeValue.textContent = `${value}٪`;
        }
    }

    setActiveView(view) {
        this.currentView = view;
        
        // آپدیت UI
        document.querySelectorAll('.view-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-view="${view}"]`)?.classList.add('active');
        
        // تغییر نمایش نتایج
        this.updateResultsView();
    }

    async startScan() {
        if (this.isScanning) return;
        
        this.isScanning = true;
        this.showScanStatus();
        
        // شبیه‌سازی اسکن
        await this.simulateScan();
        
        this.isScanning = false;
        this.hideScanStatus();
        this.loadSampleData();
    }

    async startAdvancedScan() {
        if (this.isScanning) return;
        
        this.isScanning = true;
        this.showScanStatus('در حال انجام اسکن پیشرفته...');
        
        // شبیه‌سازی اسکن پیشرفته
        await this.simulateAdvancedScan();
        
        this.isScanning = false;
        this.hideScanStatus();
        this.loadAdvancedResults();
    }

    simulateScan() {
        return new Promise(resolve => {
            let progress = 0;
            const progressBar = document.querySelector('.progress-fill');
            const statusText = document.querySelector('.status-text');
            
            const interval = setInterval(() => {
                progress += Math.random() * 15;
                if (progress >= 100) {
                    progress = 100;
                    clearInterval(interval);
                    setTimeout(resolve, 500);
                }
                
                if (progressBar) progressBar.style.width = `${progress}%`;
                if (statusText) {
                    statusText.textContent = `در حال اسکن بازار... ${Math.round(progress)}%`;
                }
            }, 200);
        });
    }

    simulateAdvancedScan() {
        return new Promise(resolve => {
            let progress = 0;
            const progressBar = document.querySelector('.progress-fill');
            const statusText = document.querySelector('.status-text');
            
            const interval = setInterval(() => {
                progress += Math.random() * 8;
                if (progress >= 100) {
                    progress = 100;
                    clearInterval(interval);
                    setTimeout(resolve, 500);
                }
                
                if (progressBar) progressBar.style.width = `${progress}%`;
                if (statusText) {
                    const steps = [
                        'در حال جمع‌آوری داده‌ها...',
                        'در حال تحلیل تکنیکال...',
                        'در حال بررسی اندیکاتورها...',
                        'در حال اعمال فیلترها...',
                        'تکمیل اسکن...'
                    ];
                    const stepIndex = Math.floor(progress / 20);
                    statusText.textContent = `${steps[stepIndex]} ${Math.round(progress)}%`;
                }
            }, 300);
        });
    }

    showScanStatus(message = 'در حال اسکن بازار...') {
        const status = document.getElementById('scanStatus');
        const statusText = document.querySelector('.status-text');
        const progressBar = document.querySelector('.progress-fill');
        
        if (status && statusText && progressBar) {
            statusText.textContent = message;
            progressBar.style.width = '0%';
            status.classList.add('active');
        }
    }

    hideScanStatus() {
        const status = document.getElementById('scanStatus');
        if (status) {
            status.classList.remove('active');
        }
    }

    loadSampleData() {
        // داده‌های نمونه برای نمایش
        this.scanResults = this.generateSampleResults();
        this.renderResults();
        this.updateStats();
    }

    loadAdvancedResults() {
        // نتایج پیشرفته
        this.scanResults = this.generateAdvancedResults();
        this.renderResults();
        this.updateStats();
    }

    generateSampleResults() {
        const symbols = ['BTC', 'ETH', 'ADA', 'SOL', 'DOT', 'LINK', 'BNB', 'XRP', 'DOGE', 'MATIC'];
        const results = [];

        symbols.forEach(symbol => {
            const price = 100 + Math.random() * 40000;
            const change = (Math.random() - 0.3) * 15;
            const volume = 50 + Math.random() * 950;
            const confidence = 70 + Math.random() * 30;
            const signalType = change > 0 ? 'bullish' : 'bearish';
            
            results.push({
                symbol: symbol,
                name: this.getCoinName(symbol),
                price: price,
                change: change,
                volume: volume,
                marketCap: volume * price * 1000,
                confidence: confidence,
                signalType: signalType,
                signalReason: this.getSignalReason(signalType, symbol)
            });
        });

        return results.sort((a, b) => b.confidence - a.confidence);
    }

    generateAdvancedResults() {
        // نتایج پیشرفته با داده‌های بیشتر
        const basicResults = this.generateSampleResults();
        return basicResults.map(result => ({
            ...result,
            rsi: 30 + Math.random() * 50,
            macd: (Math.random() - 0.5) * 3,
            volumeChange: (Math.random() - 0.3) * 100,
            trendStrength: 50 + Math.random() * 50
        }));
    }

    getCoinName(symbol) {
        const names = {
            'BTC': 'Bitcoin',
            'ETH': 'Ethereum',
            'ADA': 'Cardano',
            'SOL': 'Solana',
            'DOT': 'Polkadot',
            'LINK': 'Chainlink',
            'BNB': 'Binance Coin',
            'XRP': 'Ripple',
            'DOGE': 'Dogecoin',
            'MATIC': 'Polygon'
        };
        return names[symbol] || symbol;
    }

    getSignalReason(type, symbol) {
        const bullishReasons = [
            `شکست مقاومت در ${symbol}`,
            `الگوی صعودی در ${symbol}`,
            `حجم معاملات بالا در ${symbol}`,
            `تغییر احساسات بازار برای ${symbol}`
        ];
        
        const bearishReasons = [
            `شکست حمایت در ${symbol}`,
            `الگوی نزولی در ${symbol}`,
            `فروش شدید در ${symbol}`,
            `تغییر منفی احساسات برای ${symbol}`
        ];
        
        const reasons = type === 'bullish' ? bullishReasons : bearishReasons;
        return reasons[Math.floor(Math.random() * reasons.length)];
    }

    renderResults() {
        const container = document.getElementById('resultsGrid');
        if (!container) return;

        const filteredResults = this.filterResults();
        const sortedResults = this.sortResults(this.currentSort || 'confidence');

        container.innerHTML = sortedResults.map(result => `
            <div class="result-card ${result.signalType}" onclick="scanner.showResultDetails('${result.symbol}')">
                <div class="result-header">
                    <div class="symbol-info">
                        <div class="symbol-icon">${result.symbol.charAt(0)}</div>
                        <div class="symbol-details">
                            <h3>${result.symbol}/USDT</h3>
                            <div class="symbol-name">${result.name}</div>
                        </div>
                    </div>
                    <div class="confidence-badge">${Math.round(result.confidence)}%</div>
                </div>

                <div class="signal-type ${result.signalType}">
                    <span class="signal-icon">${result.signalType === 'bullish' ? '📈' : '📉'}</span>
                    <span>${result.signalType === 'bullish' ? 'سیگنال خرید' : 'سیگنال فروش'}</span>
                </div>

                <div class="result-stats">
                    <div class="stat-row">
                        <span class="stat-label">قیمت فعلی:</span>
                        <span class="stat-value">$${result.price.toLocaleString('en-US', {maximumFractionDigits: 2})}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">تغییر 24h:</span>
                        <span class="stat-value ${result.change >= 0 ? 'positive' : 'negative'}">
                            ${result.change >= 0 ? '+' : ''}${result.change.toFixed(2)}%
                        </span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">حجم 24h:</span>
                        <span class="stat-value">${Math.round(result.volume)}M</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">مارکت‌کپ:</span>
                        <span class="stat-value">${Math.round(result.marketCap / 1000000)}M</span>
                    </div>
                </div>

                <div class="signal-reason">
                    ${result.signalReason}
                </div>
            </div>
        `).join('');
    }

    filterResults() {
        if (this.currentFilter === 'all') {
            return this.scanResults;
        }

        return this.scanResults.filter(result => {
            switch (this.currentFilter) {
                case 'bullish':
                    return result.signalType === 'bullish';
                case 'bearish':
                    return result.signalType === 'bearish';
                case 'high-volume':
                    return result.volume > 500;
                case 'breakout':
                    return Math.abs(result.change) > 8;
                default:
                    return true;
            }
        });
    }

    sortResults(criteria) {
        this.currentSort = criteria;
        const filteredResults = this.filterResults();

        switch (criteria) {
            case 'confidence':
                return filteredResults.sort((a, b) => b.confidence - a.confidence);
            case 'volume':
                return filteredResults.sort((a, b) => b.volume - a.volume);
            case 'change':
                return filteredResults.sort((a, b) => Math.abs(b.change) - Math.abs(a.change));
            case 'marketcap':
                return filteredResults.sort((a, b) => b.marketCap - a.marketCap);
            default:
                return filteredResults;
        }
    }

    updateResultsView() {
        const container = document.getElementById('resultsGrid');
        if (!container) return;

        if (this.currentView === 'list') {
            container.style.gridTemplateColumns = '1fr';
        } else {
            container.style.gridTemplateColumns = 'repeat(auto-fill, minmax(300px, 1fr))';
        }
    }

    updateStats() {
        const totalSymbols = document.getElementById('totalSymbols');
        const signalsFound = document.getElementById('signalsFound');
        const scanTime = document.getElementById('scanTime');

        if (totalSymbols) totalSymbols.textContent = this.scanResults.length;
        if (signalsFound) signalsFound.textContent = this.scanResults.filter(r => r.confidence > 75).length;
        if (scanTime) scanTime.textContent = `${(0.5 + Math.random() * 2).toFixed(1)}s`;
    }

    showResultDetails(symbol) {
        const result = this.scanResults.find(r => r.symbol === symbol);
        if (result) {
            alert(`جزئیات ${symbol}:\n\n` +
                  `قیمت: $${result.price.toLocaleString()}\n` +
                  `تغییر: ${result.change >= 0 ? '+' : ''}${result.change.toFixed(2)}%\n` +
                  `اعتماد: ${Math.round(result.confidence)}%\n` +
                  `سیگنال: ${result.signalType === 'bullish' ? 'خرید' : 'فروش'}\n\n` +
                  `دلیل: ${result.signalReason}`);
        }
    }
}

// ایجاد instance全局
const scanner = new MarketScanner();

// راه‌اندازی
document.addEventListener('DOMContentLoaded', () => {
    // scanner already initialized
});
