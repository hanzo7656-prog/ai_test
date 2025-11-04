// سیستم اصلی VortexAI
class VortexApp {
    constructor() {
        this.currentSection = 'scan';
        this.selectedSymbols = [];
        this.scanMode = 'basic';
        this.batchSize = 25;
        this.isScanning = false;
        this.currentScan = null;
        
        // لیست کامل 100 ارز برتر
        this.top100Symbols = [
            "bitcoin", "ethereum", "tether", "ripple", "binancecoin",
            "solana", "usd-coin", "staked-ether", "tron", "dogecoin",
            "cardano", "polkadot", "chainlink", "litecoin", "bitcoin-cash",
            "stellar", "monero", "ethereum-classic", "vechain", "theta-token",
            "filecoin", "cosmos", "tezos", "aave", "eos",
            "okb", "crypto-com-chain", "algorand", "maker", "iota",
            "avalanche-2", "compound", "dash", "zcash", "neo",
            "kusama", "elrond-erd-2", "helium", "decentraland", "the-sandbox",
            "gala", "axie-infinity", "enjincoin", "render-token", "theta-fuel",
            "fantom", "klay-token", "waves", "arweave", "bittorrent",
            "huobi-token", "nexo", "celo", "qtum", "ravencoin",
            "basic-attention-token", "holotoken", "chiliz", "curve-dao-token", "kusama",
            "yearn-finance", "sushi", "uma", "balancer", "renbtc",
            "0x", "bancor", "loopring", "reserve-rights-token", "orchid",
            "nucypher", "livepeer", "api3", "uma", "badger-dao",
            "keep-network", "origin-protocol", "mirror-protocol", "radicle", "fetchtoken",
            "ocean-protocol", "dock", "request-network", "district0x", "gnosis",
            "kyber-network", "republic-protocol", "aeternity", "golem", "iostoken",
            "wax", "dent", "stormx", "funfair", "enigma",
            "singularitynet", "numeraire", "civic", "poa-network", "metal",
            "pillar", "bluzelle", "cybermiles", "datum", "edgeware"
        ];
        
        this.init();
    }

    init() {
        this.bindEvents();
        this.loadSettings();
        this.checkAPIStatus();
        this.showSection('scan');
    }

    bindEvents() {
        // Navigation
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.showSection(e.target.dataset.section);
            });
        });

        // فیلتر ارز
        document.getElementById('filterToggle').addEventListener('click', (e) => {
            e.stopPropagation();
            this.toggleFilterMenu();
        });

        document.querySelectorAll('.filter-option').forEach(option => {
            option.addEventListener('click', (e) => {
                const count = parseInt(e.target.dataset.count);
                this.selectTopSymbols(count);
                this.hideFilterMenu();
            });
        });

        // حالت اسکن
        document.querySelectorAll('input[name="scanMode"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.scanMode = e.target.value;
            });
        });

        // ورود ارزها
        document.getElementById('symbolsInput').addEventListener('input', (e) => {
            this.updateSelectedSymbols(e.target.value);
        });

        // شروع اسکن
        document.getElementById('startScan').addEventListener('click', () => {
            this.startSmartScan();
        });

        // مدیریت نتایج
        document.getElementById('clearResults').addEventListener('click', () => {
            this.clearResults();
        });

        // سلامت سیستم
        document.getElementById('refreshHealth').addEventListener('click', () => {
            this.loadHealthStatus();
        });

        // تنظیمات
        document.getElementById('saveSettings').addEventListener('click', () => {
            this.saveSettings();
        });

        document.getElementById('clearCache').addEventListener('click', () => {
            this.clearCache();
        });

        // لودینگ
        document.getElementById('cancelScan').addEventListener('click', () => {
            this.cancelScan();
        });

        // بستن منو با کلیک خارج
        document.addEventListener('click', () => {
            this.hideFilterMenu();
        });
    }

    showSection(section) {
        // آپدیت navigation
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.section === section);
        });

        // آپدیت محتوا
        document.querySelectorAll('.content-section').forEach(sect => {
            sect.classList.toggle('active', sect.id === `${section}-section`);
        });

        this.currentSection = section;

        // لود داده‌های خاص هر بخش
        switch(section) {
            case 'dashboard':
                this.loadDashboard();
                break;
            case 'health':
                this.loadHealthStatus();
                break;
            case 'settings':
                this.loadSettings();
                break;
        }
    }

    toggleFilterMenu() {
        const menu = document.getElementById('filterMenu');
        menu.classList.toggle('show');
    }

    hideFilterMenu() {
        const menu = document.getElementById('filterMenu');
        menu.classList.remove('show');
    }

    selectTopSymbols(count) {
        const topSymbols = this.top100Symbols.slice(0, count);
        this.selectedSymbols = topSymbols;
        this.updateSymbolsInput();
    }

    updateSelectedSymbols(text) {
        this.selectedSymbols = text.split('\n')
            .map(s => s.trim())
            .filter(s => s.length > 0);
        
        this.updateSelectedCount();
    }

    updateSymbolsInput() {
        const input = document.getElementById('symbolsInput');
        input.value = this.selectedSymbols.join('\n');
        this.updateSelectedCount();
    }

    updateSelectedCount() {
        const countElement = document.getElementById('selectedCount');
        if (countElement) {
            countElement.textContent = `${this.selectedSymbols.length} ارز انتخاب شده`;
        }
    }

    async startSmartScan() {
        if (this.isScanning) {
            alert('اسکن در حال انجام است!');
            return;
        }

        const symbolsToScan = this.selectedSymbols.length > 0 ? 
            this.selectedSymbols : this.top100Symbols.slice(0, 100);

        if (symbolsToScan.length === 0) {
            alert('لطفاً حداقل یک ارز انتخاب کنید');
            return;
        }

        this.isScanning = true;
        this.currentScan = new ScanSession({
            symbols: symbolsToScan,
            mode: this.scanMode,
            batchSize: this.batchSize
        });

        await this.currentScan.start();
        this.isScanning = false;
    }

    cancelScan() {
        if (this.currentScan) {
            this.currentScan.cancel();
        }
        this.hideLoading();
    }

    showLoading() {
        document.getElementById('loadingOverlay').style.display = 'flex';
    }

    hideLoading() {
        document.getElementById('loadingOverlay').style.display = 'none';
    }

    clearResults() {
        const resultsGrid = document.getElementById('resultsGrid');
        if (resultsGrid) {
            resultsGrid.innerHTML = `
                <div class="empty-state">
                    <p>نتایج پاکسازی شد</p>
                </div>
            `;
        }
        
        const resultsCount = document.getElementById('resultsCount');
        if (resultsCount) {
            resultsCount.textContent = '0 مورد';
        }
    }

    async checkAPIStatus() {
        try {
            const response = await fetch('/api/system/status');
            const data = await response.json();
            
            const statusDot = document.getElementById('statusDot');
            const statusText = document.getElementById('statusText');
            
            if (data.status === 'operational') {
                statusDot.className = 'status-dot';
                statusText.textContent = 'متصل';
            } else {
                statusDot.className = 'status-dot offline';
                statusText.textContent = 'قطع';
            }
        } catch (error) {
            console.error('خطا در بررسی وضعیت API:', error);
            const statusDot = document.getElementById('statusDot');
            const statusText = document.getElementById('statusText');
            statusDot.className = 'status-dot offline';
            statusText.textContent = 'خطا';
        }
    }

    async loadDashboard() {
        // آپدیت آمار داشبورد
        const cacheCount = document.getElementById('cacheCount');
        const totalSymbols = document.getElementById('totalSymbols');
        const apiStatus = document.getElementById('apiStatus');
        
        if (cacheCount) cacheCount.textContent = '0';
        if (totalSymbols) totalSymbols.textContent = this.top100Symbols.length;
        if (apiStatus) apiStatus.textContent = 'متصل';
    }

    async loadHealthStatus() {
        try {
            const response = await fetch('/api/system/status');
            const data = await response.json();
            
            this.displayEndpointsHealth(data.endpoints_health || {});
            this.displaySystemMetrics(data.system_metrics || {});
            this.displayLogs(data);
            
        } catch (error) {
            console.error('خطا در دریافت وضعیت سلامت:', error);
            this.displayHealthError(error);
        }
    }

    displayEndpointsHealth(endpoints) {
        const container = document.getElementById('endpointsList');
        if (!container) return;

        if (Object.keys(endpoints).length === 0) {
            container.innerHTML = '<div class="endpoint-item">داده‌ای برای نمایش موجود نیست</div>';
            return;
        }

        let html = '';
        for (const [endpoint, info] of Object.entries(endpoints)) {
            const statusClass = info.status === 'success' ? 'status-success' : 'status-error';
            const statusText = info.status === 'success' ? 'فعال' : 'خطا';
            const responseTime = info.response_time ? `${info.response_time}ms` : '--';
            const errorCode = info.error_code ? `کد: ${info.error_code}` : '';
            
            html += `
                <div class="endpoint-item">
                    <div class="endpoint-info">
                        <div class="endpoint-name">${endpoint}</div>
                        <div class="endpoint-details">
                            <span class="response-time">${responseTime}</span>
                            ${errorCode ? `<span class="error-code">${errorCode}</span>` : ''}
                        </div>
                    </div>
                    <span class="endpoint-status ${statusClass}">
                        ${statusText}
                    </span>
                </div>
            `;
        }
        container.innerHTML = html;
    }

    displaySystemMetrics(metrics) {
        const container = document.getElementById('systemMetrics');
        if (!container) return;

        container.innerHTML = `
            <div class="metric-item">مصرف CPU: ${metrics.cpu?.percent || 0}%</div>
            <div class="metric-item">مصرف RAM: ${metrics.memory?.percent || 0}%</div>
            <div class="metric-item">فضای دیسک: ${metrics.disk?.percent || 0}%</div>
            <div class="metric-item">آپتایم: ${metrics.uptime_seconds ? Math.floor(metrics.uptime_seconds / 3600) + 'h' : '--'}</div>
        `;
    }

    displayLogs(data) {
        const container = document.getElementById('logsContainer');
        if (!container) return;

        const timestamp = new Date().toLocaleString('fa-IR');
        
        let logs = `
            <div class="log-entry">
                <span class="log-time">${timestamp}</span>
                وضعیت سیستم: ${data.status || 'نامشخص'}
            </div>
        `;

        if (data.services) {
            logs += `
                <div class="log-entry">
                    <span class="log-time">${timestamp}</span>
                    سرویس CoinStats: ${data.services.coinstats_api ? 'فعال' : 'غیرفعال'}
                </div>
            `;
        }

        if (data.timestamp) {
            logs += `
                <div class="log-entry">
                    <span class="log-time">${timestamp}</span>
                    آخرین بروزرسانی: ${new Date(data.timestamp).toLocaleString('fa-IR')}
                </div>
            `;
        }

        container.innerHTML = logs;
    }

    displayHealthError(error) {
        const endpointsList = document.getElementById('endpointsList');
        const logsContainer = document.getElementById('logsContainer');
        
        if (endpointsList) {
            endpointsList.innerHTML = `
                <div class="endpoint-item error">
                    <span class="endpoint-name">خطا در دریافت داده‌های سلامت</span>
                    <span class="endpoint-status status-error">قطع</span>
                </div>
            `;
        }
        
        if (logsContainer) {
            logsContainer.innerHTML = `
                <div class="log-entry error">
                    <span class="log-time">${new Date().toLocaleString('fa-IR')}</span>
                    خطا در اتصال به API: ${error.message}
                </div>
            `;
        }
    }

    loadSettings() {
        // بارگذاری تنظیمات از localStorage
        const savedBatchSize = localStorage.getItem('vortex_batchSize') || '25';
        const savedCacheTTL = localStorage.getItem('vortex_cacheTTL') || '300';
        
        const batchSizeSelect = document.getElementById('batchSize');
        const cacheTTLSelect = document.getElementById('cacheTTL');
        
        if (batchSizeSelect) batchSizeSelect.value = savedBatchSize;
        if (cacheTTLSelect) cacheTTLSelect.value = savedCacheTTL;
        
        this.batchSize = parseInt(savedBatchSize);
    }

    saveSettings() {
        const batchSize = document.getElementById('batchSize').value;
        const cacheTTL = document.getElementById('cacheTTL').value;
        
        localStorage.setItem('vortex_batchSize', batchSize);
        localStorage.setItem('vortex_cacheTTL', cacheTTL);
        
        this.batchSize = parseInt(batchSize);
        alert('تنظیمات با موفقیت ذخیره شد');
    }

    clearCache() {
        // پاکسازی کش
        localStorage.clear();
        alert('کش سیستم با موفقیت پاکسازی شد');
    }
}

// سیستم اسکن
class ScanSession {
    constructor(options) {
        this.symbols = options.symbols;
        this.mode = options.mode;
        this.batchSize = options.batchSize;
        this.isCancelled = false;
        this.startTime = null;
        this.completed = 0;
        this.results = [];
    }

    async start() {
        this.startTime = Date.now();
        this.isCancelled = false;
        this.completed = 0;
        this.results = [];
        
        vortexApp.showLoading();
        this.updateLoadingUI();

        try {
            // تقسیم به دسته‌ها
            const batches = [];
            for (let i = 0; i < this.symbols.length; i += this.batchSize) {
                batches.push(this.symbols.slice(i, i + this.batchSize));
            }

            for (let i = 0; i < batches.length; i++) {
                if (this.isCancelled) break;

                const batch = batches[i];
                await this.processBatch(batch, i + 1, batches.length);
                
                // تاخیر بین دسته‌ها برای جلوگیری از rate limit
                if (i < batches.length - 1 && !this.isCancelled) {
                    await this.delay(500);
                }
            }

            if (!this.isCancelled) {
                this.displayResults();
                this.showCompletionMessage();
            }

        } catch (error) {
            console.error('خطا در اسکن:', error);
            this.showError('خطا در انجام اسکن: ' + error.message);
        } finally {
            vortexApp.hideLoading();
        }
    }

    async processBatch(batch, batchNumber, totalBatches) {
        const batchPromises = batch.map(symbol => this.scanSymbol(symbol));
        const batchResults = await Promise.allSettled(batchPromises);
        
        // پردازش نتایج
        const successfulResults = batchResults
            .filter(result => result.status === 'fulfilled' && result.value.success)
            .map(result => result.value);
        
        const failedResults = batchResults
            .filter(result => result.status === 'fulfilled' && !result.value.success)
            .map(result => result.value);

        this.results.push(...successfulResults, ...failedResults);
        this.completed += batch.length;

        this.updateLoadingUI(batch, batchNumber, totalBatches);
        this.displayPartialResults();
    }

    async scanSymbol(symbol) {
        try {
            const endpoint = this.mode === 'ai' ? 
                `/api/scan/ai/${symbol}` : `/api/scan/basic/${symbol}`;
            
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 10000); // تایم‌اوت 10 ثانیه
            
            const response = await fetch(endpoint, {
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            return {
                symbol,
                success: true,
                data: data,
                timestamp: new Date().toISOString()
            };

        } catch (error) {
            console.error(`خطا در اسکن ${symbol}:`, error);
            return {
                symbol,
                success: false,
                error: error.message,
                timestamp: new Date().toISOString()
            };
        }
    }

    updateLoadingUI(currentBatch = [], batchNumber = 1, totalBatches = 1) {
        const total = this.symbols.length;
        const percent = Math.round((this.completed / total) * 100);
        const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
        const speed = elapsed > 0 ? Math.round((this.completed / elapsed) * 60) : 0;

        // آپدیت UI
        const progressText = document.getElementById('progressText');
        const progressPercent = document.getElementById('progressPercent');
        const progressFill = document.getElementById('progressFill');
        const elapsedTime = document.getElementById('elapsedTime');
        const scanSpeed = document.getElementById('scanSpeed');
        const loadingTitle = document.getElementById('loadingTitle');

        if (progressText) progressText.textContent = `${this.completed}/${total}`;
        if (progressPercent) progressPercent.textContent = `${percent}%`;
        if (progressFill) progressFill.style.width = `${percent}%`;
        if (elapsedTime) elapsedTime.textContent = this.formatTime(elapsed);
        if (scanSpeed) scanSpeed.textContent = `${speed}/دقیقه`;
        if (loadingTitle) {
            loadingTitle.textContent = `اسکن ${this.mode === 'ai' ? 'داده کامل' : 'داده بهینه'} - دسته ${batchNumber}/${totalBatches}`;
        }

        // نمایش ارزهای در حال اسکن
        const scanningList = document.getElementById('scanningList');
        if (scanningList && currentBatch.length > 0) {
            scanningList.innerHTML = currentBatch
                .slice(0, 5)
                .map(symbol => `<span class="coin-tag scanning">${symbol.toUpperCase()}</span>`)
                .join('');
        }

        // نمایش ارزهای تکمیل شده
        const completedList = document.getElementById('completedList');
        if (completedList) {
            const completedSymbols = this.results
                .slice(-8)
                .map(r => r.symbol);
            
            completedList.innerHTML = completedSymbols
                .map(symbol => `<span class="coin-tag completed">${symbol.toUpperCase()}</span>`)
                .join('');
        }
    }

    displayPartialResults() {
        const container = document.getElementById('resultsGrid');
        const countElement = document.getElementById('resultsCount');
        
        if (countElement) {
            const successCount = this.results.filter(r => r.success).length;
            const totalCount = this.results.length;
            countElement.textContent = `${successCount}/${totalCount} مورد`;
        }
        
        if (container && this.results.length > 0) {
            const html = this.results.map(result => this.createCoinCard(result)).join('');
            container.innerHTML = `<div class="coin-grid">${html}</div>`;
        }
    }

    displayResults() {
        this.displayPartialResults();
    }

    createCoinCard(result) {
        if (!result.success) {
            return `
                <div class="coin-card error">
                    <div class="coin-header">
                        <div class="coin-icon">❌</div>
                        <div class="coin-basic-info">
                            <div class="coin-symbol">${result.symbol.toUpperCase()}</div>
                            <div class="coin-name">خطا در دریافت داده</div>
                        </div>
                    </div>
                    <div class="error-message">
                        ${result.error}
                    </div>
                    <div class="coin-footer">
                        <span class="data-freshness">${this.getDataFreshness(result.timestamp)}</span>
                    </div>
                </div>
            `;
        }

        const data = result.data.data || {};
        const displayData = data.display_data || {};
        const analysis = data.analysis || {};
        
        const price = displayData.price || 0;
        const change = displayData.price_change_24h || displayData.priceChange1d || 0;
        const changeClass = change >= 0 ? 'positive' : 'negative';
        const changeSymbol = change >= 0 ? '▲' : '▼';
        
        const volume = displayData.volume_24h || displayData.volume || 0;
        const marketCap = displayData.market_cap || displayData.marketCap || 0;
        const rank = displayData.rank || '--';
        
        const signal = analysis.signal || 'HOLD';
        const confidence = analysis.confidence || 0.5;
        const signalText = this.getSignalText(signal);
        const signalClass = this.getSignalClass(signal);

        return `
            <div class="coin-card">
                <div class="coin-header">
                    <div class="coin-icon">${this.getCoinSymbol(result.symbol)}</div>
                    <div class="coin-basic-info">
                        <div class="coin-symbol">${result.symbol.toUpperCase()}</div>
                        <div class="coin-name">${displayData.name || 'Unknown'}</div>
                    </div>
                    <div class="coin-price-section">
                        <div class="coin-price">$${this.formatPrice(price)}</div>
                        <div class="price-change ${changeClass}">
                            ${changeSymbol} ${Math.abs(change).toFixed(2)}%
                        </div>
                    </div>
                </div>

                <div class="coin-stats">
                    <div class="stat-item">
                        <span class="stat-label">حجم 24h</span>
                        <span class="stat-value">${this.formatNumber(volume)}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">مارکت کپ</span>
                        <span class="stat-value">${this.formatNumber(marketCap)}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">رتبه</span>
                        <span class="stat-value">#${rank}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">نوسان</span>
                        <span class="stat-value">${analysis.volatility || 0}%</span>
                    </div>
                </div>

                <div class="coin-analysis">
                    <div class="signal-badge ${signalClass}">${signalText}</div>
                    <div class="confidence-meter">
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${confidence * 100}%"></div>
                        </div>
                        <div class="confidence-text">سطح اعتماد: ${Math.round(confidence * 100)}%</div>
                    </div>
                </div>

                <div class="coin-footer">
                    <span class="data-freshness">${this.getDataFreshness(result.timestamp)}</span>
                    ${this.mode === 'ai' ? '<span class="ai-badge">AI Analysis</span>' : ''}
                </div>
            </div>
        `;
    }

    // توابع کمکی
    getCoinSymbol(symbol) {
        const symbolsMap = {
            'bitcoin': '₿',
            'ethereum': 'Ξ',
            'tether': '₮',
            'ripple': 'X',
            'binancecoin': 'BNB',
            'solana': 'SOL',
            'usd-coin': 'USDC',
            'staked-ether': 'ETH2',
            'tron': 'TRX',
            'dogecoin': 'DOGE',
            'cardano': 'ADA',
            'polkadot': 'DOT',
            'chainlink': 'LINK',
            'litecoin': 'LTC',
            'bitcoin-cash': 'BCH',
            'stellar': 'XLM',
            'monero': 'XMR',
            'ethereum-classic': 'ETC',
            'vechain': 'VET',
            'theta-token': 'THETA'
        };
        return symbolsMap[symbol] || symbol.substring(0, 3).toUpperCase();
    }

    getSignalText(signal) {
        const signals = {
            'STRONG_BUY': 'خرید قوی',
            'BUY': 'خرید',
            'HOLD': 'نگهداری',
            'SELL': 'فروش',
            'STRONG_SELL': 'فروش قوی'
        };
        return signals[signal] || signal;
    }

    getSignalClass(signal) {
        const classes = {
            'STRONG_BUY': 'signal-buy',
            'BUY': 'signal-buy',
            'HOLD': 'signal-hold',
            'SELL': 'signal-sell',
            'STRONG_SELL': 'signal-sell'
        };
        return classes[signal] || 'signal-hold';
    }

    formatPrice(price) {
        if (price === 0) return '0.00';
        if (price < 0.01) return price.toFixed(6);
        if (price < 1) return price.toFixed(4);
        if (price < 1000) return price.toFixed(2);
        return price.toLocaleString('en-US', { maximumFractionDigits: 2 });
    }

    formatNumber(num) {
        if (num === 0) return '0';
        if (num < 1000) return num.toString();
        if (num < 1000000) return (num / 1000).toFixed(1) + 'K';
        if (num < 1000000000) return (num / 1000000).toFixed(1) + 'M';
        if (num < 1000000000000) return (num / 1000000000).toFixed(1) + 'B';
        return (num / 1000000000000).toFixed(1) + 'T';
    }

    getDataFreshness(timestamp) {
        const now = new Date();
        const dataTime = new Date(timestamp);
        const diffMinutes = Math.round((now - dataTime) / (1000 * 60));
        
        if (diffMinutes < 1) return 'همین لحظه';
        if (diffMinutes < 5) return 'دقایقی پیش';
        if (diffMinutes < 30) return 'اخیراً';
        return 'قدیمی';
    }

    formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    showCompletionMessage() {
        const successCount = this.results.filter(r => r.success).length;
        const totalCount = this.results.length;
        
        if (successCount > 0) {
            console.log(`اسکن با موفقیت تکمیل شد: ${successCount}/${totalCount} ارز`);
        }
    }

    showError(message) {
        alert(message);
    }

    cancel() {
        this.isCancelled = true;
        console.log('اسکن لغو شد');
    }
}

// راه‌اندازی برنامه
const vortexApp = new VortexApp();
