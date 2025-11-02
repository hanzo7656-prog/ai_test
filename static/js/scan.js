// static/js/scan.js - کاملاً اصلاح شده
// خط اول هر فایل JS
const API_BASE_URL = 'https://ai-test-grzf.onrender.com';
class MarketScanner {
    constructor() {
        this.scanResults = [];
        this.isScanning = false;
        this.scanHistory = [];
        this.currentFilters = {
            min_confidence: 0.6,
            max_change: 15,
            volume_threshold: 1000000,
            signal_type: 'all'
        };
        this.updateInterval = null;
        
        this.initializeScanner();
        this.setupEventListeners();
    }

    async initializeScanner() {
        console.log('🚀 راه‌اندازی اسکنر بازار...');
        
        // بارگذاری تاریخچه اسکن
        this.loadScanHistory();
        
        // بارگذاری فیلترهای ذخیره شده
        this.loadSavedFilters();
        
        // به روزرسانی آمار اولیه
        this.updateStats();
        
        console.log('✅ اسکنر بازار راه‌اندازی شد');
    }

    async startScan() {
        if (this.isScanning) {
            this.showNotification('اسکن در حال انجام است...');
            return;
        }
        
        this.isScanning = true;
        this.showScanStatus('در حال اسکن بازار...');
        
        try {
            await this.performRealScan();
            this.addToScanHistory();
            
        } catch (error) {
            console.error('Scan error:', error);
            this.showScanError('خطا در انجام اسکن');
        } finally {
            this.isScanning = false;
            this.hideScanStatus();
        }
    }

    async performRealScan() {
        console.log('🔍 شروع اسکن واقعی...');
        
        const response = await fetch(`${API_BASE_URL}/api/ai/scan`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                symbols: this.getScanSymbols(),
                conditions: this.currentFilters,
                timeframe: "1h"
            })
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Scan API error: ${response.status} - ${errorText}`);
        }

        const data = await response.json();
        console.log('📊 نتایج اسکن واقعی:', data);

        if (data.status === 'success' && data.scan_results) {
            this.scanResults = data.scan_results;
            this.applyFilters();
            this.renderRealResults();
            this.updateStats();
            
            // به روزرسانی state全局
            window.appState = window.appState || {};
            window.appState.scanResults = data.scan_results;
            window.appState.lastScanTime = new Date().toISOString();
            window.appState.scanFilters = this.currentFilters;
            
        } else {
            throw new Error('نتایج اسکن معتبر نیست');
        }
    }

    getScanSymbols() {
        // لیست کامل نمادها برای اسکن - هماهنگ با دیگر فایل‌ها
        return [
            "BTC", "ETH", "SOL", "ADA", "DOT", "LINK", "BNB", "XRP", 
            "DOGE", "MATIC", "LTC", "BCH", "XLM", "ATOM", "ETC", "XMR",
            "AVAX", "TRX", "ALGO", "FTM"
        ];
    }

    applyFilters() {
        if (!this.scanResults || this.scanResults.length === 0) return;

        let filteredResults = [...this.scanResults];

        // فیلتر confidence
        filteredResults = filteredResults.filter(item => 
            (item.ai_signal?.confidence || 0) >= this.currentFilters.min_confidence
        );

        // فیلتر تغییرات قیمت
        filteredResults = filteredResults.filter(item => 
            Math.abs(item.change || 0) <= this.currentFilters.max_change
        );

        // فیلتر نوع سیگنال
        if (this.currentFilters.signal_type !== 'all') {
            filteredResults = filteredResults.filter(item => 
                item.ai_signal?.primary_signal === this.currentFilters.signal_type.toUpperCase()
            );
        }

        this.scanResults = filteredResults;
    }

    renderRealResults() {
        const container = document.getElementById('resultsGrid');
        if (!container) {
            console.warn('❌ container نتایج یافت نشد');
            return;
        }

        if (this.scanResults.length === 0) {
            container.innerHTML = `
                <div class="no-results">
                    <div class="no-results-icon">🔍</div>
                    <h3>هیچ نمادی با شرایط اسکن یافت نشد</h3>
                    <p>فیلترها را تنظیم کنید یا دوباره اسکن کنید</p>
                    <button class="btn btn-primary" onclick="scanner.startScan()">
                        اسکن مجدد
                    </button>
                </div>
            `;
            return;
        }

        container.innerHTML = this.scanResults.map(result => `
            <div class="result-card ${result.ai_signal?.primary_signal?.toLowerCase() || 'neutral'}" 
                 onclick="scanner.showResultDetails('${result.symbol}')">
                <div class="result-header">
                    <div class="symbol-info">
                        <div class="symbol-icon">${this.getSymbolIcon(result.symbol)}</div>
                        <div class="symbol-details">
                            <h3>${result.symbol}/USDT</h3>
                            <div class="symbol-name">${this.getCoinName(result.symbol)}</div>
                        </div>
                    </div>
                    <div class="confidence-badge ${this.getConfidenceLevel(result.ai_signal?.confidence)}">
                        ${Math.round((result.ai_signal?.confidence || 0) * 100)}%
                    </div>
                </div>

                <div class="signal-type ${result.ai_signal?.primary_signal?.toLowerCase() || 'neutral'}">
                    <span class="signal-icon">
                        ${this.getSignalIcon(result.ai_signal?.primary_signal)}
                    </span>
                    <span class="signal-text">
                        ${this.getSignalText(result.ai_signal?.primary_signal)}
                    </span>
                </div>

                <div class="result-stats">
                    <div class="stat-row">
                        <span class="stat-label">قیمت فعلی:</span>
                        <span class="stat-value">$${(result.current_price || 0).toLocaleString()}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">تغییر 24h:</span>
                        <span class="stat-value ${(result.change || 0) >= 0 ? 'positive' : 'negative'}">
                            ${(result.change || 0) >= 0 ? '+' : ''}${(result.change || 0).toFixed(2)}%
                        </span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">حجم معاملات:</span>
                        <span class="stat-value">${this.formatVolume(result.volume_24h)}</span>
                    </div>
                </div>

                <div class="signal-reason">
                    ${result.ai_signal?.reasoning || 'تحلیل AI پیشرفته'}
                </div>

                <div class="result-actions">
                    <button class="btn btn-sm btn-outline" onclick="event.stopPropagation(); scanner.analyzeSymbol('${result.symbol}')">
                        تحلیل
                    </button>
                    <button class="btn btn-sm btn-primary" onclick="event.stopPropagation(); scanner.addToWatchlist('${result.symbol}')">
                        پیگیری
                    </button>
                </div>
            </div>
        `).join('');
    }

    getSymbolIcon(symbol) {
        const icons = {
            'BTC': '₿', 'ETH': 'Ξ', 'SOL': '◎', 'ADA': 'A',
            'DOT': '●', 'LINK': '🔗', 'BNB': 'B', 'XRP': 'X',
            'DOGE': 'Ð', 'MATIC': 'M', 'LTC': 'Ł', 'BCH': 'B',
            'XLM': 'X', 'ATOM': '⚛', 'ETC': 'ξ', 'XMR': 'ɱ'
        };
        return icons[symbol] || symbol.charAt(0);
    }

    getCoinName(symbol) {
        const names = {
            'BTC': 'Bitcoin', 'ETH': 'Ethereum', 'SOL': 'Solana', 'ADA': 'Cardano',
            'DOT': 'Polkadot', 'LINK': 'Chainlink', 'BNB': 'Binance Coin', 
            'XRP': 'Ripple', 'DOGE': 'Dogecoin', 'MATIC': 'Polygon',
            'LTC': 'Litecoin', 'BCH': 'Bitcoin Cash', 'XLM': 'Stellar',
            'ATOM': 'Cosmos', 'ETC': 'Ethereum Classic', 'XMR': 'Monero',
            'AVAX': 'Avalanche', 'TRX': 'Tron', 'ALGO': 'Algorand', 'FTM': 'Fantom'
        };
        return names[symbol] || symbol;
    }

    getSignalIcon(signal) {
        const icons = {
            'BUY': '📈',
            'SELL': '📉', 
            'NEUTRAL': '⚪'
        };
        return icons[signal] || '⚪';
    }

    getSignalText(signal) {
        const texts = {
            'BUY': 'سیگنال خرید',
            'SELL': 'سیگنال فروش', 
            'NEUTRAL': 'خنثی'
        };
        return texts[signal] || 'در حال تحلیل';
    }

    getConfidenceLevel(confidence) {
        if (!confidence) return 'low';
        if (confidence >= 0.8) return 'high';
        if (confidence >= 0.6) return 'medium';
        return 'low';
    }

    formatVolume(volume) {
        if (!volume) return '---';
        if (volume >= 1000000000) return (volume / 1000000000).toFixed(1) + 'B';
        if (volume >= 1000000) return (volume / 1000000).toFixed(1) + 'M';
        if (volume >= 1000) return (volume / 1000).toFixed(1) + 'K';
        return volume.toFixed(0);
    }

    updateStats() {
        const totalSymbols = document.getElementById('totalSymbols');
        const signalsFound = document.getElementById('signalsFound');
        const scanTime = document.getElementById('scanTime');
        const strongSignals = document.getElementById('strongSignals');

        if (totalSymbols) totalSymbols.textContent = this.scanResults.length;
        
        const buySignals = this.scanResults.filter(item => 
            item.ai_signal?.primary_signal === 'BUY'
        ).length;
        
        const strongSignalsCount = this.scanResults.filter(item => 
            item.ai_signal && item.ai_signal.confidence > 0.7
        ).length;
        
        if (signalsFound) signalsFound.textContent = buySignals;
        if (strongSignals) strongSignals.textContent = strongSignalsCount;
        if (scanTime) scanTime.textContent = this.getScanDuration();
    }

    getScanDuration() {
        // شبیه‌سازی زمان اسکن - در واقعیت از API بگیرید
        return (1.5 + Math.random()).toFixed(1) + 's';
    }

    showResultDetails(symbol) {
        const result = this.scanResults.find(r => r.symbol === symbol);
        if (result) {
            const modalHtml = `
                <div class="modal-overlay active" onclick="scanner.closeModal()">
                    <div class="modal-content" onclick="event.stopPropagation()">
                        <div class="modal-header">
                            <h3>جزئیات ${symbol}</h3>
                            <button class="modal-close" onclick="scanner.closeModal()">×</button>
                        </div>
                        <div class="modal-body">
                            <div class="detail-section">
                                <h4>📊 اطلاعات قیمت</h4>
                                <div class="detail-grid">
                                    <div class="detail-item">
                                        <span>قیمت فعلی:</span>
                                        <span>$${(result.current_price || 0).toLocaleString()}</span>
                                    </div>
                                    <div class="detail-item">
                                        <span>تغییر 24h:</span>
                                        <span class="${(result.change || 0) >= 0 ? 'positive' : 'negative'}">
                                            ${(result.change || 0) >= 0 ? '+' : ''}${(result.change || 0).toFixed(2)}%
                                        </span>
                                    </div>
                                    <div class="detail-item">
                                        <span>حجم معاملات:</span>
                                        <span>${this.formatVolume(result.volume_24h)}</span>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="detail-section">
                                <h4>🤖 تحلیل AI</h4>
                                <div class="detail-grid">
                                    <div class="detail-item">
                                        <span>سیگنال:</span>
                                        <span class="signal ${result.ai_signal?.primary_signal?.toLowerCase()}">
                                            ${this.getSignalText(result.ai_signal?.primary_signal)}
                                        </span>
                                    </div>
                                    <div class="detail-item">
                                        <span>اعتماد:</span>
                                        <span class="confidence ${this.getConfidenceLevel(result.ai_signal?.confidence)}">
                                            ${Math.round((result.ai_signal?.confidence || 0) * 100)}%
                                        </span>
                                    </div>
                                </div>
                                <div class="reasoning">
                                    <strong>دلیل تحلیل:</strong>
                                    <p>${result.ai_signal?.reasoning || 'تحلیل AI پیشرفته'}</p>
                                </div>
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button class="btn btn-outline" onclick="scanner.closeModal()">بستن</button>
                            <button class="btn btn-primary" onclick="scanner.analyzeSymbol('${symbol}')">تحلیل پیشرفته</button>
                        </div>
                    </div>
                </div>
            `;
            
            document.body.insertAdjacentHTML('beforeend', modalHtml);
        }
    }

    closeModal() {
        const modal = document.querySelector('.modal-overlay');
        if (modal) {
            modal.remove();
        }
    }

    analyzeSymbol(symbol) {
        console.log(`🔍 تحلیل نماد: ${symbol}`);
        this.closeModal();
        window.location.href = `/analysis?symbol=${symbol}`;
    }

    addToWatchlist(symbol) {
        console.log(`⭐ افزودن به واچلیست: ${symbol}`);
        
        // ذخیره در localStorage
        const watchlist = JSON.parse(localStorage.getItem('vortex-watchlist') || '[]');
        if (!watchlist.includes(symbol)) {
            watchlist.push(symbol);
            localStorage.setItem('vortex-watchlist', JSON.stringify(watchlist));
            this.showNotification(`نماد ${symbol} به واچلیست اضافه شد`);
        } else {
            this.showNotification(`نماد ${symbol} قبلاً در واچلیست موجود است`);
        }
    }

    setupEventListeners() {
        // دکمه شروع اسکن
        document.getElementById('startScan')?.addEventListener('click', () => {
            this.startScan();
        });

        // دکمه اسکن پیشرفته
        document.getElementById('advancedScan')?.addEventListener('click', () => {
            this.showAdvancedSettings();
        });

        // فیلترها
        document.getElementById('confidenceFilter')?.addEventListener('input', (e) => {
            this.currentFilters.min_confidence = parseFloat(e.target.value) / 100;
            this.updateFilterDisplay('confidenceValue', e.target.value + '%');
            this.applyFiltersAndRender();
        });

        document.getElementById('changeFilter')?.addEventListener('input', (e) => {
            this.currentFilters.max_change = parseFloat(e.target.value);
            this.updateFilterDisplay('changeValue', e.target.value + '%');
            this.applyFiltersAndRender();
        });

        document.getElementById('signalFilter')?.addEventListener('change', (e) => {
            this.currentFilters.signal_type = e.target.value;
            this.applyFiltersAndRender();
        });

        // ذخیره فیلترها
        document.getElementById('saveFilters')?.addEventListener('click', () => {
            this.saveFilters();
        });

        // بازنشانی فیلترها
        document.getElementById('resetFilters')?.addEventListener('click', () => {
            this.resetFilters();
        });

        console.log('✅ event listenerهای اسکنر راه‌اندازی شدند');
    }

    updateFilterDisplay(elementId, value) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = value;
        }
    }

    applyFiltersAndRender() {
        if (this.scanResults.length > 0) {
            this.applyFilters();
            this.renderRealResults();
            this.updateStats();
        }
    }

    saveFilters() {
        localStorage.setItem('vortex-scan-filters', JSON.stringify(this.currentFilters));
        this.showNotification('فیلترها ذخیره شدند');
    }

    loadSavedFilters() {
        const saved = localStorage.getItem('vortex-scan-filters');
        if (saved) {
            this.currentFilters = { ...this.currentFilters, ...JSON.parse(saved) };
            this.applySavedFiltersToUI();
        }
    }

    applySavedFiltersToUI() {
        const confidenceFilter = document.getElementById('confidenceFilter');
        const changeFilter = document.getElementById('changeFilter');
        const signalFilter = document.getElementById('signalFilter');

        if (confidenceFilter) {
            confidenceFilter.value = this.currentFilters.min_confidence * 100;
            this.updateFilterDisplay('confidenceValue', Math.round(this.currentFilters.min_confidence * 100) + '%');
        }

        if (changeFilter) {
            changeFilter.value = this.currentFilters.max_change;
            this.updateFilterDisplay('changeValue', this.currentFilters.max_change + '%');
        }

        if (signalFilter) {
            signalFilter.value = this.currentFilters.signal_type;
        }
    }

    resetFilters() {
        this.currentFilters = {
            min_confidence: 0.6,
            max_change: 15,
            volume_threshold: 1000000,
            signal_type: 'all'
        };
        this.applySavedFiltersToUI();
        this.applyFiltersAndRender();
        this.showNotification('فیلترها بازنشانی شدند');
    }

    showAdvancedSettings() {
        this.showNotification('تنظیمات پیشرفته به زودی اضافه می‌شود');
    }

    loadScanHistory() {
        const history = localStorage.getItem('vortex-scan-history');
        if (history) {
            this.scanHistory = JSON.parse(history).slice(0, 10); // آخرین 10 اسکن
        }
    }

    addToScanHistory() {
        const scanRecord = {
            timestamp: new Date().toISOString(),
            resultsCount: this.scanResults.length,
            filters: { ...this.currentFilters }
        };
        
        this.scanHistory.unshift(scanRecord);
        this.scanHistory = this.scanHistory.slice(0, 10); // حفظ آخرین 10 رکورد
        
        localStorage.setItem('vortex-scan-history', JSON.stringify(this.scanHistory));
    }

    showScanStatus(message = 'در حال اسکن بازار...') {
        const status = document.getElementById('scanStatus');
        const statusText = document.querySelector('.status-text');
        const progressBar = document.querySelector('.progress-fill');
        
        if (status && statusText && progressBar) {
            statusText.textContent = message;
            progressBar.style.width = '0%';
            status.classList.add('active');
            
            // انیمیشن progress bar
            let progress = 0;
            const interval = setInterval(() => {
                progress += 2;
                progressBar.style.width = `${progress}%`;
                
                if (progress >= 100) {
                    clearInterval(interval);
                }
            }, 100);
        }
    }

    hideScanStatus() {
        const status = document.getElementById('scanStatus');
        if (status) {
            status.classList.remove('active');
        }
    }

    showScanError(message) {
        const container = document.getElementById('resultsGrid');
        if (container) {
            container.innerHTML = `
                <div class="scan-error">
                    <div class="error-icon">❌</div>
                    <h3>${message}</h3>
                    <p>لطفاً دوباره تلاش کنید یا از داده‌های نمونه استفاده کنید</p>
                    <button class="btn btn-primary" onclick="scanner.useSampleData()">
                        استفاده از داده نمونه
                    </button>
                </div>
            `;
        }
        
        this.updateStats();
    }

    useSampleData() {
        console.log('🔄 استفاده از داده‌های نمونه...');
        
        // استفاده از داده‌های global state اگر موجود باشد
        if (window.appState && window.appState.marketData) {
            this.scanResults = window.appState.marketData;
        } else {
            // داده‌های نمونه
            this.scanResults = this.generateSampleData();
        }
        
        this.applyFilters();
        this.renderRealResults();
        this.updateStats();
        this.showNotification('داده‌های نمونه بارگذاری شدند');
    }

    generateSampleData() {
        const symbols = this.getScanSymbols();
        return symbols.map(symbol => ({
            symbol: symbol,
            current_price: 1000 + Math.random() * 50000,
            change: (Math.random() - 0.5) * 20,
            volume_24h: 1000000 + Math.random() * 5000000,
            ai_signal: {
                primary_signal: Math.random() > 0.6 ? 'BUY' : Math.random() > 0.3 ? 'SELL' : 'NEUTRAL',
                confidence: 0.5 + Math.random() * 0.5,
                reasoning: 'تحلیل AI پیشرفته بر اساس الگوهای بازار'
            }
        }));
    }

    showNotification(message) {
        const notification = document.createElement('div');
        notification.className = 'scan-notification';
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: var(--accent-primary);
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            z-index: 10000;
            animation: slideInRight 0.3s ease;
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }

    // متد cleanup
    destroy() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        console.log('🧹 اسکنر بازار cleanup شد');
    }
}

// ایجاد instance جهانی
const scanner = new MarketScanner();

// راه‌اندازی
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 DOM Ready - Market Scanner Initialized');
});

// مدیریت unload صفحه
window.addEventListener('beforeunload', function() {
    if (window.scanner) {
        window.scanner.destroy();
    }
});
