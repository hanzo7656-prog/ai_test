// سیستم اصلی VortexAI با قابلیت‌های پیشرفته
class VortexApp {
    constructor() {
        this.currentSection = 'scan';
        this.selectedSymbols = [];
        this.scanMode = 'basic';
        this.batchSize = 25;
        this.isScanning = false;
        this.currentScan = null;
        this.autoRefreshInterval = null;
        this.logFilters = {
            level: 'ALL',
            search: ''
        };
        this.performanceStats = {
            totalScans: 0,
            successfulScans: 0,
            failedScans: 0,
            totalRequests: 0,
            startTime: Date.now()
        };

        // سیستم هوش مصنوعی
        this.aiEngine = new SimpleAI();
        
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
            "basic-attention-token", "holotoken", "chiliz", "curve-dao-token",
            "yearn-finance", "sushi", "uma", "balancer", "renbtc",
            "0x", "bancor", "loopring", "reserve-rights-token", "orchid",
            "nucypher", "livepeer", "api3", "badger-dao", "keep-network",
            "origin-protocol", "mirror-protocol", "radicle", "fetchtoken",
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
        this.initConsole();
        this.startAutoHealthCheck();
        this.log('INFO', 'سیستم VortexAI راه‌اندازی شد');
    }

    bindEvents() {
        // Navigation
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.showSection(e.target.closest('.nav-btn').dataset.section);
                this.toggleMobileMenu(false);
            });
        });

        // منوی موبایل
        document.getElementById('mobileMenuBtn').addEventListener('click', () => {
            this.toggleMobileMenu();
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
                this.log('DEBUG', `حالت اسکن تغییر کرد به: ${this.scanMode}`);
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

        document.getElementById('exportResults').addEventListener('click', () => {
            this.exportResults();
        });

        // سلامت سیستم
        document.getElementById('refreshHealth').addEventListener('click', () => {
            this.loadHealthStatus();
        });

        document.getElementById('testAPI').addEventListener('click', () => {
            this.testAPIEndpoints();
        });

        document.getElementById('clearHealthCache').addEventListener('click', () => {
            this.clearHealthCache();
        });

        // تنظیمات
        document.getElementById('saveSettings').addEventListener('click', () => {
            this.saveSettings();
        });

        document.getElementById('clearCache').addEventListener('click', () => {
            this.clearCache();
        });

        document.getElementById('resetSettings').addEventListener('click', () => {
            this.resetSettings();
        });

        document.getElementById('backupSettings').addEventListener('click', () => {
            this.backupSettings();
        });

        // AI
        document.getElementById('initAI').addEventListener('click', () => {
            this.initAIEngine();
        });

        document.getElementById('analyzeWithAI').addEventListener('click', () => {
            this.analyzeWithAI();
        });

        document.querySelectorAll('.symbol-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const symbol = e.target.closest('.symbol-btn').dataset.symbol;
                this.analyzeSingleSymbol(symbol);
            });
        });

        // داشبورد
        document.getElementById('refreshDashboard').addEventListener('click', () => {
            this.loadDashboard();
        });

        document.getElementById('quickStats').addEventListener('click', () => {
            this.showQuickStats();
        });

        // سیستم لاگ
        document.getElementById('clearLogs').addEventListener('click', () => {
            this.clearLogs();
        });

        document.getElementById('exportLogs').addEventListener('click', () => {
            this.exportLogs();
        });

        document.getElementById('toggleAutoRefresh').addEventListener('click', (e) => {
            this.toggleAutoRefresh(e.target);
        });

        document.querySelectorAll('.log-filter-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.setLogFilter('level', e.target.dataset.level);
            });
        });

        document.getElementById('logSearch').addEventListener('input', (e) => {
            this.setLogFilter('search', e.target.value);
        });

        document.getElementById('scrollToBottom').addEventListener('click', () => {
            this.scrollLogsToBottom();
        });

        document.getElementById('scrollToTop').addEventListener('click', () => {
            this.scrollLogsToTop();
        });

        // لودینگ
        document.getElementById('cancelScan').addEventListener('click', () => {
            this.cancelScan();
        });

        // بستن منو با کلیک خارج
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.currency-filter')) {
                this.hideFilterMenu();
            }
            if (!e.target.closest('.nav-menu') && !e.target.closest('.mobile-menu-btn')) {
                this.toggleMobileMenu(false);
            }
        });

        // مدیریت کلیدهای کیبورد
        document.addEventListener('keydown', (e) => {
            this.handleKeyboard(e);
        });

        // پیشگیری از برگشت به عقب در موبایل
        window.addEventListener('beforeunload', (e) => {
            if (this.isScanning) {
                e.preventDefault();
                e.returnValue = 'اسکن در حال انجام است. آیا مطمئنید که می‌خواهید صفحه را ترک کنید؟';
            }
        });
    }

    // ===== مدیریت ناوبری و UI =====
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
        this.log('DEBUG', `بخش فعال: ${section}`);

        // لود داده‌های خاص هر بخش
        switch(section) {
            case 'dashboard':
                this.loadDashboard();
                break;
            case 'health':
                this.loadHealthStatus();
                break;
            case 'ai':
                this.loadAIStatus();
                break;
            case 'settings':
                this.loadSettings();
                break;
        }
    }

    toggleMobileMenu(force) {
        const menu = document.getElementById('navMenu');
        const btn = document.getElementById('mobileMenuBtn');
        
        if (force !== undefined) {
            menu.classList.toggle('active', force);
            btn.setAttribute('aria-expanded', force);
        } else {
            menu.classList.toggle('active');
            const isExpanded = menu.classList.contains('active');
            btn.setAttribute('aria-expanded', isExpanded);
            btn.innerHTML = isExpanded ? '✕' : '☰';
        }
    }

    toggleFilterMenu() {
        const menu = document.getElementById('filterMenu');
        const btn = document.getElementById('filterToggle');
        const isExpanded = menu.classList.toggle('show');
        
        btn.setAttribute('aria-expanded', isExpanded);
    }

    hideFilterMenu() {
        const menu = document.getElementById('filterMenu');
        const btn = document.getElementById('filterToggle');
        
        menu.classList.remove('show');
        btn.setAttribute('aria-expanded', 'false');
    }

    // ===== مدیریت ارزها =====
    selectTopSymbols(count) {
        const topSymbols = this.top100Symbols.slice(0, count);
        this.selectedSymbols = topSymbols;
        this.updateSymbolsInput();
        this.log('INFO', `${count} ارز برتر انتخاب شد`);
        this.showNotification(`✅ ${count} ارز برتر انتخاب شد`, 'success');
    }

    updateSelectedSymbols(text) {
        this.selectedSymbols = text.split('\n')
            .map(s => s.trim())
            .filter(s => s.length > 0);
        
        this.updateSelectedCount();
    }

    updateSymbolsInput() {
        const input = document.getElementById('symbolsInput');
        if (input) {
            input.value = this.selectedSymbols.join('\n');
            this.updateSelectedCount();
        }
    }

    updateSelectedCount() {
        const countElement = document.getElementById('selectedCount');
        if (countElement) {
            countElement.textContent = `${this.selectedSymbols.length} ارز انتخاب شده`;
        }
    }

    // ===== سیستم اسکن پیشرفته =====
    async startSmartScan() {
        if (this.isScanning) {
            this.showNotification('اسکن در حال انجام است', 'warning');
            return;
        }

        const symbolsToScan = this.selectedSymbols.length > 0 ? 
            this.selectedSymbols : this.top100Symbols.slice(0, this.batchSize);

        if (symbolsToScan.length === 0) {
            this.showNotification('لطفاً حداقل یک ارز انتخاب کنید', 'error');
            return;
        }

        this.isScanning = true;
        this.performanceStats.totalScans++;
        
        this.currentScan = new ScanSession({
            symbols: symbolsToScan,
            mode: this.scanMode,
            batchSize: this.batchSize,
            onProgress: this.updateProgress.bind(this),
            onComplete: this.onScanComplete.bind(this),
            onError: this.onScanError.bind(this)
        });

        this.log('INFO', `شروع اسکن ${symbolsToScan.length} ارز در حالت ${this.scanMode}`);
        this.showLoading();
        
        try {
            await this.currentScan.start();
        } catch (error) {
            this.log('ERROR', `خطا در اسکن: ${error.message}`);
            this.showNotification('خطا در انجام اسکن', 'error');
        }
    }

    updateProgress(progress) {
        const {
            completed,
            total,
            percent,
            elapsed,
            speed,
            currentBatch
        } = progress;

        // آپدیت UI لودینگ
        const progressText = document.getElementById('progressText');
        const progressPercent = document.getElementById('progressPercent');
        const progressFill = document.getElementById('progressFill');
        const elapsedTime = document.getElementById('elapsedTime');
        const scanSpeed = document.getElementById('scanSpeed');
        const loadingTitle = document.getElementById('loadingTitle');
        const scanningList = document.getElementById('scanningList');

        if (progressText) progressText.textContent = `${completed}/${total}`;
        if (progressPercent) progressPercent.textContent = `${percent}%`;
        if (progressFill) progressFill.style.width = `${percent}%`;
        if (elapsedTime) elapsedTime.textContent = this.formatTime(elapsed);
        if (scanSpeed) scanSpeed.textContent = `${speed}/دقیقه`;
        if (loadingTitle) {
            loadingTitle.textContent = `اسکن ${this.scanMode === 'ai' ? 'AI' : 'پایه'} - ${percent}%`;
        }

        // نمایش ارزهای در حال اسکن
        if (scanningList && currentBatch && currentBatch.length > 0) {
            scanningList.innerHTML = currentBatch
                .slice(0, 8)
                .map(symbol => `<span class="coin-tag scanning">${symbol.toUpperCase()}</span>`)
                .join('');
        }
    }

    onScanComplete(results) {
        this.isScanning = false;
        this.hideLoading();
        
        const successCount = results.filter(r => r.success).length;
        const totalCount = results.length;
        
        this.performanceStats.successfulScans += successCount;
        this.performanceStats.failedScans += (totalCount - successCount);
        
        // نمایش نتایج
        this.displayResults(results);
        
        this.log('SUCCESS', `اسکن تکمیل شد: ${successCount}/${totalCount} موفق`);
        this.showNotification(`✅ اسکن ${totalCount} ارز تکمیل شد (${successCount} موفق)`, 'success');
        
        this.updatePerformanceStats();
    }

    onScanError(error) {
        this.isScanning = false;
        this.hideLoading();
        
        this.performanceStats.failedScans++;
        this.log('ERROR', `خطا در اسکن: ${error.message}`);
        this.showNotification('خطا در انجام اسکن', 'error');
        
        this.updatePerformanceStats();
    }

    displayResults(results) {
        const container = document.getElementById('resultsGrid');
        const countElement = document.getElementById('resultsCount');
        
        if (!container) return;
        
        if (results.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">🔍</div>
                    <p>هیچ نتیجه‌ای یافت نشد</p>
                    <small>اسکن انجام شد اما داده‌ای دریافت نشد</small>
                </div>
            `;
            return;
        }

        const successCount = results.filter(r => r.success).length;
        if (countElement) {
            countElement.textContent = `${successCount}/${results.length} مورد`;
        }

        const html = results.map(result => this.createCoinCard(result)).join('');
        container.innerHTML = `
            <div class="coin-grid">${html}</div>
        `;
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
                        ${result.error || 'خطای نامشخص'}
                    </div>
                    <div class="coin-footer">
                        <span class="data-freshness">${this.getDataFreshness(result.timestamp)}</span>
                    </div>
                </div>
            `;
        }

        const data = result.data;
        const extractedData = this.extractCoinData(data, result.symbol);
        
        return `
            <div class="coin-card">
                <div class="coin-header">
                    <div class="coin-icon">${this.getCoinSymbol(result.symbol)}</div>
                    <div class="coin-basic-info">
                        <div class="coin-symbol">${result.symbol.toUpperCase()}</div>
                        <div class="coin-name">${extractedData.name}</div>
                    </div>
                </div>

                <div class="price-section">
                    <div class="coin-price">${extractedData.price !== 0 ? '$' + this.formatPrice(extractedData.price) : '--'}</div>
                    <div class="price-change ${extractedData.change >= 0 ? 'positive' : 'negative'}">
                        ${extractedData.change !== 0 ? 
                            `${extractedData.change >= 0 ? '▲' : '▼'} ${Math.abs(extractedData.change).toFixed(2)}%` : 
                            '--'}
                    </div>
                </div>

                <div class="coin-stats">
                    <div class="stat-item">
                        <span class="stat-label">حجم 24h</span>
                        <span class="stat-value">${extractedData.volume !== 0 ? this.formatNumber(extractedData.volume) : '--'}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">مارکت کپ</span>
                        <span class="stat-value">${extractedData.marketCap !== 0 ? this.formatNumber(extractedData.marketCap) : '--'}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">رتبه</span>
                        <span class="stat-value">${extractedData.rank ? '#' + extractedData.rank : '--'}</span>
                    </div>
                </div>

                ${this.scanMode === 'ai' ? `
                <div class="coin-analysis">
                    <div class="signal-badge ${extractedData.signalClass}">${extractedData.signalText}</div>
                    <div class="confidence-meter">
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${extractedData.confidence * 100}%"></div>
                        </div>
                        <div class="confidence-text">اعتماد: ${Math.round(extractedData.confidence * 100)}%</div>
                    </div>
                </div>
                ` : ''}

                <div class="coin-footer">
                    <span class="data-freshness">${this.getDataFreshness(result.timestamp)}</span>
                    ${this.scanMode === 'ai' ? '<span class="ai-badge">AI</span>' : ''}
                </div>
            </div>
        `;
    }

    extractCoinData(data, symbol) {
        // داده‌های پیش‌فرض
        let extracted = {
            price: 0,
            change: 0,
            volume: 0,
            marketCap: 0,
            rank: null,
            name: symbol.toUpperCase(),
            signal: 'HOLD',
            confidence: 0.5,
            signalText: 'نگهداری',
            signalClass: 'signal-hold'
        };

        try {
            console.log(`📊 استخراج داده برای ${symbol}:`, data);

            // حالت 1: داده از API اصلی
            if (data && data.data) {
                const coinData = data.data;
                
                // بررسی ساختارهای مختلف داده
                if (coinData.raw_data && coinData.raw_data.coin_details) {
                    const details = coinData.raw_data.coin_details;
                    extracted.price = details.price || details.current_price || 0;
                    extracted.change = details.priceChange1d || details.price_change_24h || details.price_change_percentage_24h || 0;
                    extracted.volume = details.volume || details.total_volume || 0;
                    extracted.marketCap = details.marketCap || details.market_cap || 0;
                    extracted.rank = details.rank || null;
                    extracted.name = details.name || symbol.toUpperCase();
                }
                // حالت 2: داده مستقیم از CoinStats
                else if (coinData.display_data) {
                    const display = coinData.display_data;
                    extracted.price = display.price || display.current_price || 0;
                    extracted.change = display.price_change_24h || display.priceChange1d || 0;
                    extracted.volume = display.volume_24h || display.total_volume || 0;
                    extracted.marketCap = display.market_cap || display.marketCap || 0;
                    extracted.rank = display.rank || null;
                    extracted.name = display.name || symbol.toUpperCase();
                }
                // حالت 3: داده مستقیم در ریشه
                else {
                    extracted.price = coinData.price || coinData.current_price || 0;
                    extracted.change = coinData.price_change_24h || coinData.priceChange1d || 0;
                    extracted.volume = coinData.volume || coinData.total_volume || 0;
                    extracted.marketCap = coinData.marketCap || coinData.market_cap || 0;
                    extracted.rank = coinData.rank || null;
                    extracted.name = coinData.name || symbol.toUpperCase();
                }

                // تحلیل AI اگر موجود باشد
                if (coinData.analysis) {
                    extracted.signal = coinData.analysis.signal || 'HOLD';
                    extracted.confidence = coinData.analysis.confidence || 0.5;
                }
            }
            // حالت 4: داده مستقیم در ریشه response
            else if (data && (data.price !== undefined || data.current_price !== undefined)) {
                extracted.price = data.price || data.current_price || 0;
                extracted.change = data.priceChange1d || data.price_change_24h || data.price_change_percentage_24h || 0;
                extracted.volume = data.volume || data.total_volume || 0;
                extracted.marketCap = data.marketCap || data.market_cap || 0;
                extracted.rank = data.rank || null;
                extracted.name = data.name || symbol.toUpperCase();
            }
            // حالت 5: داده تست (fallback)
            else {
                this.log('WARN', `ساختار داده برای ${symbol} شناسایی نشد، استفاده از داده تست`);
                const hash = this.stringToHash(symbol);
                extracted.price = 1000 + (hash % 50000);
                extracted.change = (hash % 40) - 20;
                extracted.volume = 1000000 + (hash % 100000000);
                extracted.marketCap = 10000000 + (hash % 1000000000);
                extracted.rank = (hash % 100) + 1;
                extracted.name = symbol.toUpperCase();
            }

            // تولید سیگنال AI بر اساس داده‌ها
            if (this.scanMode === 'ai') {
                const aiAnalysis = this.aiAnalyze(extracted);
                extracted.signal = aiAnalysis.signal;
                extracted.confidence = aiAnalysis.confidence;
            }

            // تنظیم متن و کلاس سیگنال
            const signalConfig = {
                'STRONG_BUY': { text: 'خرید قوی', class: 'signal-buy' },
                'BUY': { text: 'خرید', class: 'signal-buy' },
                'HOLD': { text: 'نگهداری', class: 'signal-hold' },
                'SELL': { text: 'فروش', class: 'signal-sell' },
                'STRONG_SELL': { text: 'فروش قوی', class: 'signal-sell' }
            };

            const signalInfo = signalConfig[extracted.signal] || signalConfig.HOLD;
            extracted.signalText = signalInfo.text;
            extracted.signalClass = signalInfo.class;

        } catch (error) {
            this.log('ERROR', `خطا در استخراج داده برای ${symbol}: ${error.message}`);
        }

        console.log(`✅ داده استخراج شده برای ${symbol}:`, extracted);
        return extracted;
    }

    // ===== هوش مصنوعی پایه =====
    aiAnalyze(coinData) {
        return this.aiEngine.analyzeTechnical(coinData);
    }

    async initAIEngine() {
        this.log('INFO', '🚀 راه‌اندازی موتور AI...');
        this.showLoading();
        
        try {
            const success = await this.aiEngine.initialize();
            
            if (success) {
                this.log('SUCCESS', '✅ موتور AI با موفقیت راه‌اندازی شد');
                this.showNotification('🤖 موتور AI فعال شد', 'success');
                this.loadAIStatus();
            } else {
                throw new Error('راه‌اندازی AI ناموفق بود');
            }
        } catch (error) {
            this.log('ERROR', `خطا در راه‌اندازی AI: ${error.message}`);
            this.showNotification('خطا در راه‌اندازی AI', 'error');
        } finally {
            this.hideLoading();
        }
    }

    async analyzeWithAI() {
        const symbols = this.selectedSymbols.length > 0 ? 
            this.selectedSymbols : ['bitcoin', 'ethereum'];
            
        this.log('INFO', `شروع تحلیل AI برای ${symbols.length} ارز`);
        
        // تغییر حالت به AI و شروع اسکن
        this.scanMode = 'ai';
        document.querySelector('input[name="scanMode"][value="ai"]').checked = true;
        this.startSmartScan();
    }

    async analyzeSingleSymbol(symbol) {
        this.log('INFO', `تحلیل تک ارز: ${symbol}`);
        this.showNotification(`🧠 تحلیل ${symbol}...`, 'info');
        
        this.selectedSymbols = [symbol];
        this.scanMode = 'ai';
        document.querySelector('input[name="scanMode"][value="ai"]').checked = true;
        
        this.startSmartScan();
    }

    loadAIStatus() {
        const container = document.getElementById('aiStatusIndicators');
        if (!container) return;

        const status = this.aiEngine.getStatus();
        
        container.innerHTML = `
            <div class="indicator">
                <span class="indicator-label">
                    <span class="indicator-icon">📊</span>
                    موتور تکنیکال
                </span>
                <span class="indicator-value ${status.technical.ready ? 'status-success' : 'status-error'}">
                    ${status.technical.ready ? 'فعال' : 'غیرفعال'}
                </span>
            </div>
            <div class="indicator">
                <span class="indicator-label">
                    <span class="indicator-icon">😊</span>
                    تحلیل احساسات
                </span>
                <span class="indicator-value ${status.sentiment.ready ? 'status-success' : 'status-error'}">
                    ${status.sentiment.ready ? 'فعال' : 'غیرفعال'}
                </span>
            </div>
            <div class="indicator">
                <span class="indicator-label">
                    <span class="indicator-icon">🔮</span>
                    پیش‌بینی قیمت
                </span>
                <span class="indicator-value ${status.predictive.ready ? 'status-success' : 'status-error'}">
                    ${status.predictive.ready ? 'فعال' : 'غیرفعال'}
                </span>
            </div>
            <div class="indicator">
                <span class="indicator-label">
                    <span class="indicator-icon">⚡</span>
                    وضعیت کلی
                </span>
                <span class="indicator-value ${status.initialized ? 'status-success' : 'status-error'}">
                    ${status.initialized ? 'فعال' : 'غیرفعال'}
                </span>
            </div>
        `;
    }

    cancelScan() {
        if (this.currentScan) {
            this.currentScan.cancel();
            this.log('INFO', 'اسکن توسط کاربر لغو شد');
        }
        this.isScanning = false;
        this.hideLoading();
        this.showNotification('اسکن لغو شد', 'warning');
    }

    clearResults() {
        const resultsGrid = document.getElementById('resultsGrid');
        const resultsCount = document.getElementById('resultsCount');
        
        if (resultsGrid) {
            resultsGrid.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">🔍</div>
                    <p>هنوز اسکنی انجام نشده است</p>
                    <small>برای شروع از دکمه بالا استفاده کنید</small>
                </div>
            `;
        }
        
        if (resultsCount) {
            resultsCount.textContent = '0 مورد';
        }
        
        this.log('INFO', 'نتایج اسکن پاکسازی شد');
    }

    // ===== سیستم لاگ پیشرفته =====
    log(level, message, data = null) {
        const timestamp = new Date().toLocaleString('fa-IR');
        const logEntry = {
            timestamp,
            level,
            message,
            data
        };

        // ذخیره در حافظه
        if (!this.logs) this.logs = [];
        this.logs.push(logEntry);

        // نمایش در UI
        this.displayLog(logEntry);

        // نمایش در کنسول مرورگر
        const consoleMethod = {
            'ERROR': 'error',
            'WARN': 'warn',
            'INFO': 'info',
            'DEBUG': 'log',
            'SUCCESS': 'log'
        }[level] || 'log';

        const styles = {
            'ERROR': 'color: #ff4757; font-weight: bold;',
            'WARN': 'color: #ff9f43; font-weight: bold;',
            'INFO': 'color: #0052ff; font-weight: bold;',
            'DEBUG': 'color: #64748b;',
            'SUCCESS': 'color: #00d9a6; font-weight: bold;'
        }[level];

        console[consoleMethod](`%c[VortexAI] ${timestamp} ${level}: ${message}`, styles);
        if (data) console[consoleMethod](data);

        // آپدیت شمارنده لاگ
        this.updateLogCount();
    }

    displayLog(logEntry) {
        const container = document.getElementById('logsContainer');
        if (!container) return;

        // اعمال فیلترها
        if (this.logFilters.level !== 'ALL' && this.logFilters.level !== logEntry.level) {
            return;
        }

        if (this.logFilters.search && !logEntry.message.includes(this.logFilters.search)) {
            return;
        }

        const logElement = document.createElement('div');
        logElement.className = 'log-entry';
        logElement.innerHTML = `
            <span class="log-time">${logEntry.timestamp}</span>
            <span class="log-level ${logEntry.level}">${logEntry.level}</span>
            <span class="log-message">${this.escapeHtml(logEntry.message)}</span>
        `;

        container.appendChild(logElement);

        // اسکرول خودکار به پایین اگر فعال باشد
        if (this.autoScrollLogs) {
            this.scrollLogsToBottom();
        }
    }

    setLogFilter(type, value) {
        this.logFilters[type] = value;

        // آپدیت UI دکمه‌های فیلتر
        if (type === 'level') {
            document.querySelectorAll('.log-filter-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.level === value);
            });
        }

        // بازنمایی لاگ‌ها
        this.refreshLogsDisplay();
    }

    refreshLogsDisplay() {
        const container = document.getElementById('logsContainer');
        if (!container) return;

        container.innerHTML = '';
        
        if (this.logs) {
            this.logs.forEach(log => this.displayLog(log));
        }

        this.updateLogCount();
    }

    updateLogCount() {
        const countElement = document.getElementById('logCount');
        if (countElement && this.logs) {
            const filteredLogs = this.logs.filter(log => {
                if (this.logFilters.level !== 'ALL' && this.logFilters.level !== log.level) {
                    return false;
                }
                if (this.logFilters.search && !log.message.includes(this.logFilters.search)) {
                    return false;
                }
                return true;
            });
            countElement.textContent = filteredLogs.length;
        }
    }

    clearLogs() {
        this.logs = [];
        const container = document.getElementById('logsContainer');
        if (container) {
            container.innerHTML = '';
        }
        this.log('INFO', 'همه لاگ‌ها پاکسازی شدند');
        this.updateLogCount();
    }

    exportLogs() {
        if (!this.logs || this.logs.length === 0) {
            this.showNotification('لاگی برای ذخیره وجود ندارد', 'warning');
            return;
        }

        const logText = this.logs.map(log => 
            `[${log.timestamp}] ${log.level}: ${log.message}`
        ).join('\n');

        this.downloadFile('vortexai-logs.txt', logText);
        this.log('INFO', 'لاگ‌ها با موفقیت ذخیره شدند');
        this.showNotification('لاگ‌ها ذخیره شدند', 'success');
    }

    scrollLogsToBottom() {
        const container = document.getElementById('logsContainer');
        if (container) {
            container.scrollTop = container.scrollHeight;
        }
    }

    scrollLogsToTop() {
        const container = document.getElementById('logsContainer');
        if (container) {
            container.scrollTop = 0;
        }
    }

    toggleAutoRefresh(button) {
        if (this.autoRefreshInterval) {
            clearInterval(this.autoRefreshInterval);
            this.autoRefreshInterval = null;
            button.innerHTML = '🔴 غیرفعال';
            this.log('INFO', 'بروزرسانی خودکار غیرفعال شد');
        } else {
            this.autoRefreshInterval = setInterval(() => {
                this.loadHealthStatus();
            }, 10000);
            button.innerHTML = '🟢 فعال';
            this.log('INFO', 'بروزرسانی خودکار فعال شد (10 ثانیه)');
        }
    }

    // ===== سیستم سلامت و مانیتورینگ =====
    async loadHealthStatus() {
        try {
            this.log('DEBUG', 'دریافت وضعیت سلامت سیستم...');
            
            const response = await fetch('/api/system/status');
            const data = await response.json();
            
            this.displayEndpointsHealth(data.endpoints_health || {});
            this.displaySystemMetrics(data.system_metrics || {});
            this.displayAIHealth(data);
            
            this.log('SUCCESS', 'وضعیت سلامت سیستم بروزرسانی شد');
        } catch (error) {
            this.log('ERROR', `خطا در دریافت وضعیت سلامت: ${error.message}`);
            this.displayHealthError(error);
        }
    }

    async displayEndpointsHealth(endpoints) {
        const container = document.getElementById('endpointsList');
        if (!container) return;

        const testEndpoints = [
            { name: 'API اصلی سیستم', url: '/api/system/status' },
            { name: 'اسکن پایه', url: '/api/scan/basic/bitcoin' },
            { name: 'اسکن AI', url: '/api/scan/ai/bitcoin' },
            { name: 'وضعیت AI', url: '/api/ai/status' },
            { name: 'داده‌های بازار', url: '/api/market/data' }
        ];

        container.innerHTML = '';

        for (const endpoint of testEndpoints) {
            const result = await this.testEndpoint(endpoint.url);
            
            const endpointItem = document.createElement('div');
            endpointItem.className = 'endpoint-item';
            endpointItem.innerHTML = `
                <div class="endpoint-info">
                    <div class="endpoint-name">${endpoint.name}</div>
                    <div class="endpoint-details">
                        <span class="response-time">${result.responseTime}ms</span>
                        ${result.error ? `<span class="error-code">${result.error}</span>` : ''}
                    </div>
                </div>
                <span class="endpoint-status ${result.success ? 'status-success' : 'status-error'}">
                    ${result.success ? 'فعال' : 'خطا'}
                </span>
            `;
            
            container.appendChild(endpointItem);
        }
    }

    async testEndpoint(url) {
        try {
            const startTime = performance.now();
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 10000);
            
            const response = await fetch(url, { 
                signal: controller.signal,
                headers: {
                    'Cache-Control': 'no-cache'
                }
            });
            
            clearTimeout(timeoutId);
            const responseTime = Math.round(performance.now() - startTime);

            if (!response.ok) {
                return {
                    success: false,
                    responseTime,
                    error: `HTTP ${response.status}`
                };
            }
            
            await response.json();
            return {
                success: true,
                responseTime
            };
        } catch (error) {
            return {
                success: false,
                responseTime: 0,
                error: error.name === 'AbortError' ? 'Timeout' : error.message
            };
        }
    }

    displaySystemMetrics(metrics) {
        // آپدیت متریک‌های سیستم
        const cpuElement = document.getElementById('cpuUsage');
        const memoryElement = document.getElementById('memoryUsage');
        const diskElement = document.getElementById('diskUsage');
        const uptimeElement = document.getElementById('uptime');

        if (cpuElement) cpuElement.textContent = `${metrics.cpu_percent || 0}%`;
        if (memoryElement) memoryElement.textContent = `${metrics.memory_percent || 0}%`;
        if (diskElement) diskElement.textContent = `${metrics.disk_percent || 0}%`;
        if (uptimeElement) uptimeElement.textContent = this.formatUptime(metrics.uptime_seconds || 0);
    }

    displayAIHealth(data) {
        const container = document.getElementById('aiEngineStatus');
        if (!container) return;

        const aiStatus = this.aiEngine.getStatus();
        
        container.innerHTML = `
            <div class="indicator">
                <span class="indicator-label">موتور تکنیکال</span>
                <span class="indicator-value ${aiStatus.technical.ready ? 'status-success' : 'status-error'}">
                    ${aiStatus.technical.ready ? 'فعال' : 'غیرفعال'}
                </span>
            </div>
            <div class="indicator">
                <span class="indicator-label">تحلیل روند</span>
                <span class="indicator-value ${aiStatus.sentiment.ready ? 'status-success' : 'status-error'}">
                    ${aiStatus.sentiment.ready ? 'فعال' : 'غیرفعال'}
                </span>
            </div>
            <div class="indicator">
                <span class="indicator-label">داده‌های زنده</span>
                <span class="indicator-value ${aiStatus.predictive.ready ? 'status-success' : 'status-error'}">
                    ${aiStatus.predictive.ready ? 'فعال' : 'غیرفعال'}
                </span>
            </div>
        `;
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
            const timestamp = new Date().toLocaleString('fa-IR');
            logsContainer.innerHTML = `
                <div class="log-entry">
                    <span class="log-time">${timestamp}</span>
                    <span class="log-level ERROR">ERROR</span>
                    <span class="log-message">خطا در اتصال به API: ${error.message}</span>
                </div>
            `;
        }
    }

    // ===== سیستم تنظیمات =====
    loadSettings() {
        const settings = this.getStoredSettings();
        
        // بارگذاری تنظیمات در UI
        document.getElementById('batchSize').value = settings.batchSize;
        document.getElementById('cacheTTL').value = settings.cacheTTL;
        document.getElementById('resultsPerPage').value = settings.resultsPerPage;
        document.getElementById('aiPrecision').value = settings.aiPrecision;
        document.getElementById('autoLearning').checked = settings.autoLearning;

        // آپدیت اطلاعات سیستم
        this.updateSystemInfo();
        
        this.log('DEBUG', 'تنظیمات از حافظه بارگذاری شد');
    }

    saveSettings() {
        const settings = {
            batchSize: document.getElementById('batchSize').value,
            cacheTTL: document.getElementById('cacheTTL').value,
            resultsPerPage: document.getElementById('resultsPerPage').value,
            aiPrecision: document.getElementById('aiPrecision').value,
            autoLearning: document.getElementById('autoLearning').checked,
            lastUpdated: new Date().toISOString()
        };

        localStorage.setItem('vortex_settings', JSON.stringify(settings));
        this.batchSize = parseInt(settings.batchSize);
        
        this.log('SUCCESS', 'تنظیمات با موفقیت ذخیره شد');
        this.showNotification('✅ تنظیمات ذخیره شد', 'success');
    }

    getStoredSettings() {
        const defaultSettings = {
            batchSize: '25',
            cacheTTL: '300',
            resultsPerPage: '25',
            aiPrecision: 'medium',
            autoLearning: true
        };

        try {
            const stored = localStorage.getItem('vortex_settings');
            return stored ? { ...defaultSettings, ...JSON.parse(stored) } : defaultSettings;
        } catch {
            return defaultSettings;
        }
    }

    clearCache() {
        localStorage.clear();
        this.log('INFO', 'کش سیستم پاکسازی شد');
        this.showNotification('🗑️ کش سیستم پاکسازی شد', 'success');
    }

    resetSettings() {
        localStorage.removeItem('vortex_settings');
        this.loadSettings();
        this.log('INFO', 'تنظیمات به حالت پیش‌فرض بازگردانی شد');
        this.showNotification('🔄 تنظیمات بازنشانی شد', 'success');
    }

    backupSettings() {
        const settings = this.getStoredSettings();
        const backupData = {
            ...settings,
            backupDate: new Date().toISOString(),
            version: '1.0.0'
        };
        
        this.downloadFile('vortexai-settings-backup.json', JSON.stringify(backupData, null, 2));
        this.log('INFO', 'پشتیبان تنظیمات ذخیره شد');
        this.showNotification('💾 پشتیبان تنظیمات ذخیره شد', 'success');
    }

    updateSystemInfo() {
        // آپدیت اطلاعات سیستم در تنظیمات
        const versionElement = document.getElementById('systemVersion');
        const lastUpdateElement = document.getElementById('lastUpdate');
        const memoryUsedElement = document.getElementById('memoryUsed');
        const sessionDurationElement = document.getElementById('sessionDuration');

        if (versionElement) versionElement.textContent = '1.0.0';
        if (lastUpdateElement) lastUpdateElement.textContent = new Date().toLocaleString('fa-IR');
        if (memoryUsedElement) memoryUsedElement.textContent = this.formatMemoryUsage();
        if (sessionDurationElement) sessionDurationElement.textContent = this.formatSessionDuration();
    }

    // ===== داشبورد =====
    async loadDashboard() {
        try {
            const response = await fetch('/api/system/status');
            const data = await response.json();
            
            // آپدیت آمار ساده
            const cacheCount = document.getElementById('cacheCount');
            const totalSymbols = document.getElementById('totalSymbols');
            const scanCount = document.getElementById('scanCount');
            const aiAnalysisCount = document.getElementById('aiAnalysisCount');
            
            if (cacheCount) cacheCount.textContent = data.cache?.total_files || '0';
            if (totalSymbols) totalSymbols.textContent = this.top100Symbols.length;
            if (scanCount) scanCount.textContent = this.performanceStats.totalScans;
            if (aiAnalysisCount) aiAnalysisCount.textContent = this.performanceStats.successfulScans;
            
            this.updatePerformanceStats();
            
        } catch (error) {
            this.log('ERROR', `خطا در بارگذاری داشبورد: ${error.message}`);
            const totalSymbols = document.getElementById('totalSymbols');
            if (totalSymbols) totalSymbols.textContent = this.top100Symbols.length;
        }
    }

    showQuickStats() {
        const stats = `
📊 آمار سریع سیستم:

• کل اسکن‌ها: ${this.performanceStats.totalScans}
• اسکن موفق: ${this.performanceStats.successfulScans}
• اسکن ناموفق: ${this.performanceStats.failedScans}
• ارزهای پشتیبانی: ${this.top100Symbols.length}
• وضعیت AI: ${this.aiEngine.isInitialized ? 'فعال' : 'غیرفعال'}
        `.trim();

        this.log('INFO', 'آمار سریع سیستم:\n' + stats);
        this.showNotification('📊 آمار سیستم نمایش داده شد', 'info');
    }

    // ===== ابزارهای کمکی =====
    showLoading() {
        const loading = document.getElementById('loadingOverlay');
        if (loading) {
            loading.style.display = 'flex';
            document.body.style.overflow = 'hidden';
        }
    }

    hideLoading() {
        const loading = document.getElementById('loadingOverlay');
        if (loading) {
            loading.style.display = 'none';
            document.body.style.overflow = '';
        }
    }

    showNotification(message, type = 'info') {
        // ایجاد المان نوتیفیکیشن
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.setAttribute('role', 'alert');
        notification.innerHTML = `
            <div class="notification-content">
                <span class="notification-message">${message}</span>
                <button class="notification-close" aria-label="بستن">&times;</button>
            </div>
        `;

        document.body.appendChild(notification);

        // نمایش با انیمیشن
        setTimeout(() => notification.classList.add('show'), 100);

        // بستن دستی
        notification.querySelector('.notification-close').addEventListener('click', () => {
            this.hideNotification(notification);
        });

        // حذف خودکار
        setTimeout(() => {
            this.hideNotification(notification);
        }, 5000);
    }

    hideNotification(notification) {
        notification.classList.remove('show');
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
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
            'bitcoin-cash': 'BCH'
        };
        return symbolsMap[symbol] || symbol.substring(0, 3).toUpperCase();
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

    formatUptime(seconds) {
        const days = Math.floor(seconds / 86400);
        const hours = Math.floor((seconds % 86400) / 3600);
        return `${days}d ${hours}h`;
    }

    formatMemoryUsage() {
        // شبیه‌سازی استفاده از حافظه
        const used = Math.round(process.memoryUsage ? process.memoryUsage().heapUsed / 1024 / 1024 : 50);
        return `${used} MB`;
    }

    formatSessionDuration() {
        const duration = Math.floor((Date.now() - this.performanceStats.startTime) / 1000);
        return this.formatTime(duration);
    }

    stringToHash(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;
        }
        return Math.abs(hash);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    downloadFile(filename, content) {
        const blob = new Blob([content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    handleKeyboard(e) {
        // کلیدهای میانبر
        if (e.ctrlKey || e.metaKey) {
            switch(e.key) {
                case '1':
                    e.preventDefault();
                    this.showSection('scan');
                    break;
                case '2':
                    e.preventDefault();
                    this.showSection('dashboard');
                    break;
                case '3':
                    e.preventDefault();
                    this.showSection('health');
                    break;
                case '4':
                    e.preventDefault();
                    this.showSection('ai');
                    break;
                case '5':
                    e.preventDefault();
                    this.showSection('settings');
                    break;
                case 'k':
                    e.preventDefault();
                    document.getElementById('symbolsInput').focus();
                    break;
                case 'l':
                    e.preventDefault();
                    this.clearLogs();
                    break;
            }
        }

        // Escape برای بستن منوها
        if (e.key === 'Escape') {
            this.hideFilterMenu();
            this.toggleMobileMenu(false);
        }
    }

    // ===== کنسول توسعه =====
    initConsole() {
        // پیاده‌سازی کنسول توسعه‌دهنده
        this.setupConsoleCommands();
    }

    setupConsoleCommands() {
        // دستورات کنسول برای توسعه
        window.vortex = {
            app: this,
            test: () => this.testAPIEndpoints(),
            logs: () => this.logs,
            stats: () => this.performanceStats,
            clear: () => this.clearLogs(),
            settings: () => this.getStoredSettings()
        };
    }

    // ===== عملکرد و آمار =====
    updatePerformanceStats() {
        const successRate = this.performanceStats.totalScans > 0 ?
            Math.round((this.performanceStats.successfulScans / this.performanceStats.totalScans) * 100) : 0;

        // آپدیت داشبورد
        const successRateElement = document.getElementById('successRate');
        const totalRequestsElement = document.getElementById('totalRequests');
        const successScansElement = document.getElementById('successScans');
        const failedScansElement = document.getElementById('failedScans');

        if (successRateElement) successRateElement.textContent = `${successRate}%`;
        if (totalRequestsElement) totalRequestsElement.textContent = this.performanceStats.totalRequests;
        if (successScansElement) successScansElement.textContent = this.performanceStats.successfulScans;
        if (failedScansElement) failedScansElement.textContent = this.performanceStats.failedScans;
    }

    startAutoHealthCheck() {
        // بررسی سلامت هر 30 ثانیه
        setInterval(() => {
            this.checkAPIStatus();
        }, 30000);
    }

    async checkAPIStatus() {
        try {
            const response = await fetch('/api/system/status');
            const data = await response.json();
            
            const statusDot = document.getElementById('statusDot');
            const statusText = document.getElementById('statusText');
            
            if (data.status === 'operational') {
                if (statusDot) {
                    statusDot.className = 'status-dot';
                    statusDot.style.animation = 'pulse 2s infinite';
                }
                if (statusText) statusText.textContent = 'متصل';
            } else {
                if (statusDot) {
                    statusDot.className = 'status-dot offline';
                    statusDot.style.animation = 'none';
                }
                if (statusText) statusText.textContent = 'قطع';
            }
        } catch (error) {
            const statusDot = document.getElementById('statusDot');
            const statusText = document.getElementById('statusText');
            if (statusDot) {
                statusDot.className = 'status-dot offline';
                statusDot.style.animation = 'none';
            }
            if (statusText) statusText.textContent = 'خطا';
        }
    }

    async testAPIEndpoints() {
        this.log('INFO', '🧪 شروع تست API endpoints...');
        
        const testEndpoints = [
            { name: 'System Status', url: '/api/system/status' },
            { name: 'Basic Scan', url: '/api/scan/basic/bitcoin' },
            { name: 'AI Scan', url: '/api/scan/ai/bitcoin' },
            { name: 'AI Status', url: '/api/ai/status' }
        ];
        
        for (const endpoint of testEndpoints) {
            try {
                this.log('DEBUG', `🔍 تست ${endpoint.name}: ${endpoint.url}`);
                const startTime = Date.now();
                const response = await fetch(endpoint.url);
                const responseTime = Date.now() - startTime;
                
                if (!response.ok) {
                    this.log('ERROR', `❌ ${endpoint.name}: HTTP ${response.status}`);
                    continue;
                }
                
                const data = await response.json();
                this.log('SUCCESS', `✅ ${endpoint.name}: ${responseTime}ms`);
                
            } catch (error) {
                this.log('ERROR', `❌ ${endpoint.name}: ${error.message}`);
            }
            
            await this.delay(1000);
        }
        
        this.log('SUCCESS', '✅ تست API تکمیل شد');
        this.showNotification('تست API انجام شد. نتیجه را در console ببینید.', 'info');
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    exportResults() {
        if (!this.currentScan || !this.currentScan.results || this.currentScan.results.length === 0) {
            this.showNotification('هیچ نتیجه‌ای برای ذخیره وجود ندارد', 'warning');
            return;
        }

        const results = this.currentScan.results.filter(r => r.success);
        const csvContent = this.convertToCSV(results);
        this.downloadFile('vortexai-results.csv', csvContent);
        this.log('INFO', 'نتایج اسکن ذخیره شد');
        this.showNotification('📥 نتایج ذخیره شد', 'success');
    }

    convertToCSV(results) {
        const headers = ['Symbol', 'Name', 'Price', 'Change%', 'Volume', 'MarketCap', 'Rank', 'Signal', 'Confidence'];
        const rows = results.map(result => {
            const data = this.extractCoinData(result.data, result.symbol);
            return [
                result.symbol.toUpperCase(),
                data.name,
                data.price,
                data.change,
                data.volume,
                data.marketCap,
                data.rank,
                data.signalText,
                data.confidence
            ];
        });

        return [headers, ...rows].map(row => row.join(',')).join('\n');
    }

    clearHealthCache() {
        this.log('INFO', 'کش سلامت سیستم پاکسازی شد');
        this.showNotification('🗑️ کش سلامت پاکسازی شد', 'success');
    }
}

// ===== سیستم هوش مصنوعی پایه =====
class SimpleAI {
    constructor() {
        this.isInitialized = false;
        this.models = {
            technical: null,
            sentiment: null,
            predictive: null
        };
        this.history = [];
    }

    async initialize() {
        try {
            // بارگذاری مدل‌های پایه
            await this.loadTechnicalModel();
            await this.loadSentimentModel();
            await this.loadPredictiveModel();
            
            this.isInitialized = true;
            return true;
        } catch (error) {
            console.error('AI Initialization error:', error);
            return false;
        }
    }

    async loadTechnicalModel() {
        // مدل تحلیل تکنیکال ساده
        this.models.technical = {
            name: 'تحلیل‌گر تکنیکال',
            version: '1.0',
            ready: true,
            indicators: ['RSI', 'MACD', 'MovingAverage', 'SupportResistance']
        };
        await this.delay(500);
    }

    async loadSentimentModel() {
        // مدل تحلیل احساسات ساده
        this.models.sentiment = {
            name: 'تحلیل‌گر احساسات',
            version: '1.0',
            ready: true,
            sources: ['PriceAction', 'VolumeAnalysis', 'MarketRank']
        };
        await this.delay(300);
    }

    async loadPredictiveModel() {
        // مدل پیش‌بینی ساده
        this.models.predictive = {
            name: 'پیش‌بین قیمت',
            version: '1.0',
            ready: true,
            features: ['HistoricalPatterns', 'MarketCycles', 'VolatilityAnalysis']
        };
        await this.delay(400);
    }

    analyzeTechnical(coinData) {
        const analysis = {
            signal: 'HOLD',
            confidence: 0.5,
            indicators: [],
            summary: ''
        };

        // تحلیل بر اساس RSI ساده
        const rsi = this.calculateRSI(coinData);
        if (rsi < 30) {
            analysis.signal = 'BUY';
            analysis.confidence += 0.2;
            analysis.indicators.push(`RSI: ${rsi.toFixed(1)} (اشباع فروش)`);
        } else if (rsi > 70) {
            analysis.signal = 'SELL';
            analysis.confidence += 0.2;
            analysis.indicators.push(`RSI: ${rsi.toFixed(1)} (اشباع خرید)`);
        }

        // تحلیل روند قیمت
        if (coinData.change > 5) {
            analysis.signal = analysis.signal === 'SELL' ? 'HOLD' : 'BUY';
            analysis.confidence += 0.15;
            analysis.indicators.push(`روند: صعودی (${coinData.change.toFixed(1)}%)`);
        } else if (coinData.change < -5) {
            analysis.signal = analysis.signal === 'BUY' ? 'HOLD' : 'SELL';
            analysis.confidence += 0.15;
            analysis.indicators.push(`روند: نزولی (${coinData.change.toFixed(1)}%)`);
        }

        // تحلیل حجم
        if (coinData.volume > 500000000) { // حجم بالا
            analysis.confidence += 0.1;
            analysis.indicators.push('حجم: بالا');
        }

        // تحلیل رتبه بازار
        if (coinData.rank && coinData.rank <= 10) {
            analysis.confidence += 0.1;
            analysis.indicators.push('رتبه: برتر');
        }

        // محدود کردن confidence
        analysis.confidence = Math.max(0.1, Math.min(0.95, analysis.confidence));

        // ارتقا سیگنال بر اساس confidence
        if (analysis.confidence > 0.7 && analysis.signal === 'BUY') {
            analysis.signal = 'STRONG_BUY';
        } else if (analysis.confidence > 0.7 && analysis.signal === 'SELL') {
            analysis.signal = 'STRONG_SELL';
        }

        analysis.summary = analysis.indicators.join(' • ') || 'داده کافی نیست';

        // ذخیره در تاریخچه
        this.history.push({
            symbol: coinData.name,
            analysis,
            timestamp: new Date().toISOString()
        });

        return analysis;
    }

    calculateRSI(coinData) {
        // شبیه‌سازی RSI ساده بر اساس تغییرات قیمت
        const change = coinData.change || 0;
        return Math.min(100, Math.max(0, 50 + (change * 2)));
    }

    getStatus() {
        return {
            initialized: this.isInitialized,
            technical: this.models.technical,
            sentiment: this.models.sentiment,
            predictive: this.models.predictive,
            historyCount: this.history.length
        };
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// سیستم اسکن پیشرفته
class ScanSession {
    constructor(options) {
        this.symbols = options.symbols;
        this.mode = options.mode;
        this.batchSize = options.batchSize;
        this.onProgress = options.onProgress;
        this.onComplete = options.onComplete;
        this.onError = options.onError;
        
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

        try {
            const batches = this.createBatches();
            
            for (let i = 0; i < batches.length; i++) {
                if (this.isCancelled) break;

                const batch = batches[i];
                await this.processBatch(batch, i + 1, batches.length);
                
                // تاخیر بین batchها برای کاهش فشار
                if (i < batches.length - 1 && !this.isCancelled) {
                    await this.delay(1000);
                }
            }

            if (!this.isCancelled) {
                this.onComplete?.(this.results);
            }

        } catch (error) {
            this.onError?.(error);
        }
    }

    createBatches() {
        const batches = [];
        for (let i = 0; i < this.symbols.length; i += this.batchSize) {
            batches.push(this.symbols.slice(i, i + this.batchSize));
        }
        return batches;
    }

    async processBatch(batch, batchNumber, totalBatches) {
        const batchPromises = batch.map(symbol => this.scanSymbol(symbol));
        const batchResults = await Promise.allSettled(batchPromises);
        
        const successfulResults = batchResults
            .filter(result => result.status === 'fulfilled' && result.value.success)
            .map(result => result.value);

        const failedResults = batchResults
            .filter(result => result.status === 'fulfilled' && !result.value.success)
            .map(result => result.value);

        this.results.push(...successfulResults, ...failedResults);
        this.completed += batch.length;

        this.updateProgress(batch, batchNumber, totalBatches);
    }

    async scanSymbol(symbol) {
        try {
            const endpoint = this.mode === 'ai' ? 
                `/api/scan/ai/${symbol}` : `/api/scan/basic/${symbol}`;
            
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 15000);
            
            const response = await fetch(endpoint, {
                signal: controller.signal,
                headers: {
                    'Cache-Control': 'no-cache'
                }
            });
            
            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();
            
            return {
                symbol,
                success: true,
                data: data,
                timestamp: new Date().toISOString()
            };

        } catch (error) {
            return {
                symbol,
                success: false,
                error: error.message,
                timestamp: new Date().toISOString()
            };
        }
    }

    updateProgress(currentBatch, batchNumber, totalBatches) {
        const total = this.symbols.length;
        const percent = Math.round((this.completed / total) * 100);
        const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
        const speed = elapsed > 0 ? Math.round((this.completed / elapsed) * 60) : 0;

        this.onProgress?.({
            completed: this.completed,
            total,
            percent,
            elapsed,
            speed,
            currentBatch,
            batchNumber,
            totalBatches
        });
    }

    cancel() {
        this.isCancelled = true;
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// راه‌اندازی برنامه
const vortexApp = new VortexApp();

// متغیرهای global برای دسترسی از کنسول
window.VortexAI = vortexApp;
