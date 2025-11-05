// سیستم اصلی VortexAI با Integration کامل Trading AI
class VortexApp {
    constructor() {
        this.currentSection = 'scan';
        this.selectedSymbols = [];
        this.scanMode = 'basic';
        this.batchSize = 25;
        this.isScanning = false;
        this.currentScan = null;
        this.scanCount = 0;
        this.aiAnalysisCount = 0;
        
        // آمار سیستم
        this.systemStats = {
            cacheItems: 0,
            todayScans: 0,
            aiAnalyses: 0,
            apiCalls: 0,
            lastUpdate: new Date()
        };

        // وضعیت AI
        this.aiStatus = {
            initialized: false,
            engineReady: false,
            lastAnalysis: null,
            error: null
        };

        // لاگ‌های سیستم
        this.systemLogs = [];
        
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
        this.checkAIStatus();
        this.showSection('scan');
        this.logSystem('سیستم VortexAI راه‌اندازی شد', 'success');
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
                this.logSystem(`حالت اسکن تغییر کرد به: ${this.scanMode === 'ai' ? 'تحلیل AI' : 'داده بهینه'}`);
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

        document.getElementById('exportData').addEventListener('click', () => {
            this.exportData();
        });

        // سلامت سیستم
        document.getElementById('refreshHealth').addEventListener('click', () => {
            this.loadHealthStatus();
        });

        document.getElementById('clearLogs').addEventListener('click', () => {
            this.clearLogs();
        });

        // داشبورد
        document.getElementById('refreshDashboard').addEventListener('click', () => {
            this.loadDashboard();
        });

        // هوش مصنوعی
        document.getElementById('initAI').addEventListener('click', () => {
            this.initializeAI();
        });

        document.getElementById('analyzeWithAI').addEventListener('click', () => {
            this.analyzeSelectedWithAI();
        });

        document.querySelectorAll('.symbol-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const symbol = e.target.dataset.symbol;
                this.analyzeSymbolWithAI(symbol);
            });
        });

        // لودینگ و مدال
        document.getElementById('cancelScan').addEventListener('click', () => {
            this.cancelScan();
        });

        document.getElementById('closeModal').addEventListener('click', () => {
            this.hideModal();
        });

        // بستن منو با کلیک خارج
        document.addEventListener('click', () => {
            this.hideFilterMenu();
        });

        // بستن مدال با کلیک خارج
        document.getElementById('aiModal').addEventListener('click', (e) => {
            if (e.target.id === 'aiModal') {
                this.hideModal();
            }
        });
    }

    // ==================== NAVIGATION ====================

    showSection(section) {
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.section === section);
        });

        document.querySelectorAll('.content-section').forEach(sect => {
            sect.classList.toggle('active', sect.id === `${section}-section`);
        });

        this.currentSection = section;

        switch(section) {
            case 'dashboard':
                this.loadDashboard();
                break;
            case 'health':
                this.loadHealthStatus();
                break;
            case 'ai':
                this.loadAISection();
                break;
        }
    }

    // ==================== SCAN SYSTEM ====================

    async startSmartScan() {
        if (this.isScanning) {
            alert('اسکن در حال انجام است!');
            return;
        }

        const symbolsToScan = this.selectedSymbols.length > 0 ? 
            this.selectedSymbols : this.top100Symbols.slice(0, 50);

        if (symbolsToScan.length === 0) {
            alert('لطفاً حداقل یک ارز انتخاب کنید');
            return;
        }

        this.isScanning = true;
        this.scanCount++;
        this.systemStats.todayScans++;

        this.currentScan = new ScanSession({
            symbols: symbolsToScan,
            mode: this.scanMode,
            batchSize: this.batchSize,
            onProgress: this.updateScanProgress.bind(this),
            onComplete: this.onScanComplete.bind(this),
            onError: this.onScanError.bind(this)
        });

        this.showGlassLoading('در حال اسکن بازار', `اسکن ${symbolsToScan.length} ارز`);
        
        try {
            await this.currentScan.start();
        } catch (error) {
            this.onScanError(error);
        }
    }

    updateScanProgress(progress) {
        this.updateGlassLoading(
            `اسکن ${progress.mode === 'ai' ? 'تحلیل AI' : 'داده بهینه'}`,
            `دسته ${progress.batch}/${progress.totalBatches} - ${progress.completed}/${progress.total} ارز`,
            (progress.completed / progress.total) * 100
        );
    }

    onScanComplete(results) {
        this.isScanning = false;
        this.hideGlassLoading();
        
        const successCount = results.filter(r => r.success).length;
        this.logSystem(`اسکن تکمیل شد: ${successCount}/${results.length} ارز موفق`, 'success');
        
        this.displayResults(results);
        this.updateDashboard();
    }

    onScanError(error) {
        this.isScanning = false;
        this.hideGlassLoading();
        this.logSystem(`خطا در اسکن: ${error.message}`, 'error');
        alert(`خطا در اسکن: ${error.message}`);
    }

    cancelScan() {
        if (this.currentScan) {
            this.currentScan.cancel();
        }
        this.isScanning = false;
        this.hideGlassLoading();
        this.logSystem('اسکن توسط کاربر لغو شد', 'warning');
    }

    // ==================== AI INTEGRATION ====================

    async initializeAI() {
        this.showGlassLoading('راه‌اندازی هوش مصنوعی', 'در حال بارگذاری موتورهای تحلیل...');
        
        try {
            const response = await fetch('/api/ai/initialize', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            });
            
            const result = await response.json();
            
            if (result.status === 'success' && result.initialized) {
                this.aiStatus.initialized = true;
                this.aiStatus.engineReady = true;
                this.aiStatus.error = null;
                
                this.hideGlassLoading();
                this.logSystem('سیستم هوش مصنوعی با موفقیت راه‌اندازی شد', 'success');
                this.updateAISection();
                alert('✅ سیستم AI آماده است!');
            } else {
                throw new Error(result.message || 'خطا در راه‌اندازی AI');
            }
            
        } catch (error) {
            this.hideGlassLoading();
            this.aiStatus.error = error.message;
            this.logSystem(`خطا در راه‌اندازی AI: ${error.message}`, 'error');
            alert(`❌ خطا در راه‌اندازی AI: ${error.message}`);
        }
    }

    async checkAIStatus() {
        try {
            const response = await fetch('/api/ai/status');
            const result = await response.json();
            
            if (result.status === 'success') {
                this.aiStatus.initialized = result.ai_system.initialized || false;
                this.aiStatus.engineReady = result.ai_system.initialized || false;
                this.aiStatus.lastAnalysis = result.ai_system.last_analysis_time;
            }
        } catch (error) {
            console.error('خطا در بررسی وضعیت AI:', error);
            this.aiStatus.error = error.message;
        }
        
        this.updateAISection();
    }

    async analyzeSymbolWithAI(symbol) {
        if (!this.aiStatus.initialized) {
            const shouldInitialize = confirm('سیستم AI راه‌اندازی نشده است. آیا می‌خواهید راه‌اندازی شود؟');
            if (shouldInitialize) {
                await this.initializeAI();
                if (!this.aiStatus.initialized) return;
            } else {
                return;
            }
        }

        this.showGlassLoading('تحلیل AI', `در حال تحلیل ${symbol}...`);
        
        try {
            const analysis = await this.callAIAnalysis(symbol);
            this.hideGlassLoading();
            
            this.aiAnalysisCount++;
            this.systemStats.aiAnalyses++;
            
            this.displayAIAnalysis(symbol, analysis);
            this.logSystem(`تحلیل AI برای ${symbol} تکمیل شد`, 'success');
            
        } catch (error) {
            this.hideGlassLoading();
            this.logSystem(`خطا در تحلیل AI برای ${symbol}: ${error.message}`, 'error');
            alert(`خطا در تحلیل AI: ${error.message}`);
        }
    }

    async analyzeSelectedWithAI() {
        const symbolsToAnalyze = this.selectedSymbols.length > 0 ? 
            this.selectedSymbols : ['bitcoin', 'ethereum'];
            
        if (!this.aiStatus.initialized) {
            await this.initializeAI();
            if (!this.aiStatus.initialized) return;
        }

        this.showGlassLoading('تحلیل دسته‌ای AI', `تحلیل ${symbolsToAnalyze.length} ارز...`);
        
        try {
            const analyses = [];
            for (const symbol of symbolsToAnalyze.slice(0, 5)) { // حداکثر 5 ارز
                const analysis = await this.callAIAnalysis(symbol);
                analyses.push({ symbol, analysis });
                
                this.updateGlassLoading(
                    'تحلیل دسته‌ای AI',
                    `در حال تحلیل ${symbol} (${analyses.length}/${Math.min(symbolsToAnalyze.length, 5)})`,
                    (analyses.length / Math.min(symbolsToAnalyze.length, 5)) * 100
                );
            }
            
            this.hideGlassLoading();
            this.aiAnalysisCount += analyses.length;
            this.systemStats.aiAnalyses += analyses.length;
            
            this.displayAIAnalyses(analyses);
            this.logSystem(`تحلیل دسته‌ای AI تکمیل شد: ${analyses.length} ارز`, 'success');
            
        } catch (error) {
            this.hideGlassLoading();
            this.logSystem(`خطا در تحلیل دسته‌ای AI: ${error.message}`, 'error');
        }
    }

    async callAIAnalysis(symbol) {
        const response = await fetch('/api/ai/analyze', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                symbol: symbol,
                analysis_type: 'comprehensive'
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        if (result.status !== 'success') {
            throw new Error(result.detail || 'خطا در تحلیل AI');
        }
        
        return result.data;
    }

    // ==================== DISPLAY SYSTEMS ====================

    displayResults(results) {
        const container = document.getElementById('resultsGrid');
        const countElement = document.getElementById('resultsCount');
        
        const successCount = results.filter(r => r.success).length;
        if (countElement) {
            countElement.textContent = `${successCount}/${results.length} مورد`;
        }
        
        if (container && results.length > 0) {
            const html = results.map(result => this.createCoinCard(result)).join('');
            container.innerHTML = `<div class="coins-grid">${html}</div>`;
        }
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
                </div>
            `;
        }

        const data = result.data;
        const extracted = this.extractCoinData(data, result.symbol, result.mode);
        
        const price = extracted.price;
        const change = extracted.change;
        const changeClass = change >= 0 ? 'positive' : 'negative';
        const changeSymbol = change >= 0 ? '▲' : '▼';

        return `
            <div class="coin-card" onclick="vortexApp.showCoinDetails('${result.symbol}', ${JSON.stringify(extracted).replace(/'/g, "\\'")})">
                <div class="coin-header">
                    <div class="coin-icon">${this.getCoinSymbol(result.symbol)}</div>
                    <div class="coin-basic-info">
                        <div class="coin-symbol">${result.symbol.toUpperCase()}</div>
                        <div class="coin-name">${extracted.name}</div>
                    </div>
                </div>

                <div class="coin-price-section">
                    <div class="coin-price">${price !== 0 ? '$' + this.formatPrice(price) : '--'}</div>
                    <div class="price-change ${changeClass}">
                        ${change !== 0 ? `${changeSymbol} ${Math.abs(change).toFixed(2)}%` : '--'}
                    </div>
                </div>

                <div class="coin-stats">
                    <div class="stat-item">
                        <span class="stat-label">حجم</span>
                        <span class="stat-value">${extracted.volume !== 0 ? this.formatNumber(extracted.volume) : '--'}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">رتبه</span>
                        <span class="stat-value">${extracted.rank ? '#' + extracted.rank : '--'}</span>
                    </div>
                </div>

                <div class="coin-analysis">
                    ${extracted.signal ? `<div class="signal-badge ${extracted.signalClass}">${extracted.signalText}</div>` : ''}
                    ${result.mode === 'ai' ? `
                        <div class="raw-data-indicator">
                            <span class="raw-dot"></span>
                            <span>تحلیل AI خام</span>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    }

    displayAIAnalysis(symbol, analysis) {
        const modalTitle = document.getElementById('modalTitle');
        const modalBody = document.getElementById('modalBody');
        
        modalTitle.textContent = `تحلیل AI - ${symbol.toUpperCase()}`;
        
        const signal = analysis.trading_signal;
        const technical = analysis.technical_analysis;
        
        modalBody.innerHTML = `
            <div class="ai-analysis-detail">
                <div class="analysis-summary">
                    <div class="signal-card ${signal.action.toLowerCase()}">
                        <h4>سیگنال معاملاتی</h4>
                        <div class="signal-main">${this.getSignalText(signal.action)}</div>
                        <div class="signal-confidence">اعتماد: ${Math.round(signal.confidence * 100)}%</div>
                        <div class="signal-reasoning">${signal.reasoning}</div>
                    </div>
                </div>
                
                <div class="analysis-details">
                    <h4>جزئیات فنی</h4>
                    <div class="detail-grid">
                        <div class="detail-item">
                            <span>قیمت فعلی:</span>
                            <span>$${this.formatPrice(technical.current_price)}</span>
                        </div>
                        <div class="detail-item">
                            <span>تغییر 24h:</span>
                            <span class="${technical.price_change_24h >= 0 ? 'positive' : 'negative'}">
                                ${technical.price_change_24h >= 0 ? '▲' : '▼'} ${Math.abs(technical.price_change_24h).toFixed(2)}%
                            </span>
                        </div>
                        <div class="detail-item">
                            <span>روند:</span>
                            <span>${technical.trend_analysis?.direction || 'نامشخص'}</span>
                        </div>
                        <div class="detail-item">
                            <span>حمایت:</span>
                            <span>$${this.formatPrice(technical.key_levels?.support)}</span>
                        </div>
                        <div class="detail-item">
                            <span>مقاومت:</span>
                            <span>$${this.formatPrice(technical.key_levels?.resistance)}</span>
                        </div>
                    </div>
                </div>
                
                ${analysis.recommendations ? `
                    <div class="recommendations">
                        <h4>توصیه‌ها</h4>
                        <ul>
                            ${analysis.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                        </ul>
                    </div>
                ` : ''}
                
                <div class="raw-data-info">
                    <h4>اطلاعات داده خام</h4>
                    <div class="raw-stats">
                        <div>نقاط داده تاریخی: ${analysis.raw_data_metrics?.historical_points || 0}</div>
                        <div>منابع داده: ${analysis.raw_data_metrics?.current_data_sources || 0}</div>
                        <div>کیفیت داده: ${analysis.raw_data_metrics?.data_quality || 'نامشخص'}</div>
                    </div>
                </div>
            </div>
        `;
        
        this.showModal();
    }

    displayAIAnalyses(analyses) {
        const container = document.getElementById('aiResults');
        const html = analyses.map(item => this.createAICoinCard(item.symbol, item.analysis)).join('');
        container.innerHTML = `<div class="coins-grid">${html}</div>`;
    }

    createAICoinCard(symbol, analysis) {
        const signal = analysis.trading_signal;
        const technical = analysis.technical_analysis;
        
        return `
            <div class="coin-card ai-card" onclick="vortexApp.showAIDetails('${symbol}', ${JSON.stringify(analysis).replace(/'/g, "\\'")})">
                <div class="coin-header">
                    <div class="coin-icon">${this.getCoinSymbol(symbol)}</div>
                    <div class="coin-basic-info">
                        <div class="coin-symbol">${symbol.toUpperCase()}</div>
                        <div class="coin-name">تحلیل AI</div>
                    </div>
                </div>

                <div class="coin-price-section">
                    <div class="coin-price">$${this.formatPrice(technical.current_price)}</div>
                    <div class="price-change ${technical.price_change_24h >= 0 ? 'positive' : 'negative'}">
                        ${technical.price_change_24h >= 0 ? '▲' : '▼'} ${Math.abs(technical.price_change_24h).toFixed(2)}%
                    </div>
                </div>

                <div class="coin-analysis">
                    <div class="signal-badge ${this.getSignalClass(signal.action)}">
                        ${this.getSignalText(signal.action)}
                    </div>
                    <div class="raw-data-indicator">
                        <span class="raw-dot"></span>
                        <span>AI خام - ${Math.round(signal.confidence * 100)}%</span>
                    </div>
                </div>
            </div>
        `;
    }

    // ==================== UTILITIES ====================

    extractCoinData(data, symbol, mode = 'basic') {
        let extracted = {
            price: 0,
            change: 0,
            volume: 0,
            marketCap: 0,
            rank: null,
            name: symbol.toUpperCase(),
            signal: null,
            signalText: null,
            signalClass: null
        };

        try {
            if (mode === 'ai' && data.technical_analysis) {
                // داده‌های AI
                const tech = data.technical_analysis;
                const signal = data.trading_signal;
                
                extracted.price = tech.current_price || 0;
                extracted.change = tech.price_change_24h || 0;
                extracted.volume = tech.indicators?.volume || 0;
                extracted.name = symbol.toUpperCase();
                extracted.signal = signal.action;
                extracted.signalText = this.getSignalText(signal.action);
                extracted.signalClass = this.getSignalClass(signal.action);
                
            } else if (data.data && data.data.display_data) {
                // داده‌های معمولی
                const displayData = data.data.display_data;
                extracted.price = displayData.price || 0;
                extracted.change = displayData.price_change_24h || displayData.priceChange1d || 0;
                extracted.volume = displayData.volume_24h || displayData.volume || 0;
                extracted.marketCap = displayData.market_cap || displayData.marketCap || 0;
                extracted.rank = displayData.rank || null;
                extracted.name = displayData.name || symbol.toUpperCase();
                
                if (data.data.analysis) {
                    extracted.signal = data.data.analysis.signal;
                    extracted.signalText = this.getSignalText(data.data.analysis.signal);
                    extracted.signalClass = this.getSignalClass(data.data.analysis.signal);
                }
            }
        } catch (error) {
            console.error('خطا در استخراج داده:', error);
        }

        return extracted;
    }

    getCoinSymbol(symbol) {
        const symbolsMap = {
            'bitcoin': '₿', 'ethereum': 'Ξ', 'tether': '₮', 'ripple': 'X',
            'binancecoin': 'BNB', 'solana': 'SOL', 'usd-coin': 'USDC',
            'cardano': 'ADA', 'dogecoin': 'Ð', 'polkadot': 'DOT',
            'chainlink': '●', 'litecoin': 'Ł', 'bitcoin-cash': 'BCH'
        };
        return symbolsMap[symbol] || symbol.substring(0, 3).toUpperCase();
    }

    getSignalText(signal) {
        const signals = {
            'STRONG_BUY': 'خرید قوی', 'BUY': 'خرید', 'HOLD': 'نگهداری',
            'SELL': 'فروش', 'STRONG_SELL': 'فروش قوی'
        };
        return signals[signal] || signal;
    }

    getSignalClass(signal) {
        const classes = {
            'STRONG_BUY': 'signal-buy', 'BUY': 'signal-buy',
            'HOLD': 'signal-hold', 'SELL': 'signal-sell', 'STRONG_SELL': 'signal-sell'
        };
        return classes[signal] || 'signal-hold';
    }

    formatPrice(price) {
        if (price === 0 || price === null) return '0.00';
        if (price < 0.01) return price.toFixed(6);
        if (price < 1) return price.toFixed(4);
        if (price < 1000) return price.toFixed(2);
        return price.toLocaleString('en-US', { maximumFractionDigits: 2 });
    }

    formatNumber(num) {
        if (num === 0 || num === null) return '0';
        if (num < 1000) return num.toString();
        if (num < 1000000) return (num / 1000).toFixed(1) + 'K';
        if (num < 1000000000) return (num / 1000000).toFixed(1) + 'M';
        return (num / 1000000000).toFixed(1) + 'B';
    }

    // ==================== UI MANAGEMENT ====================

    showGlassLoading(title, message, progress = 0) {
        const loading = document.getElementById('glassLoading');
        const titleEl = document.getElementById('loadingTitle');
        const messageEl = document.getElementById('loadingMessage');
        const progressEl = document.getElementById('loadingProgress');
        const fillEl = document.getElementById('loadingFill');
        
        titleEl.textContent = title;
        messageEl.textContent = message;
        progressEl.textContent = `${Math.round(progress)}%`;
        fillEl.style.width = `${progress}%`;
        
        loading.style.display = 'flex';
    }

    updateGlassLoading(title, message, progress) {
        const titleEl = document.getElementById('loadingTitle');
        const messageEl = document.getElementById('loadingMessage');
        const progressEl = document.getElementById('loadingProgress');
        const fillEl = document.getElementById('loadingFill');
        
        if (title) titleEl.textContent = title;
        if (message) messageEl.textContent = message;
        if (progress !== undefined) {
            progressEl.textContent = `${Math.round(progress)}%`;
            fillEl.style.width = `${progress}%`;
        }
    }

    hideGlassLoading() {
        document.getElementById('glassLoading').style.display = 'none';
    }

    showModal() {
        document.getElementById('aiModal').style.display = 'flex';
    }

    hideModal() {
        document.getElementById('aiModal').style.display = 'none';
    }

    showCoinDetails(symbol, data) {
        // نمایش جزئیات کوین در مدال
        const modalTitle = document.getElementById('modalTitle');
        const modalBody = document.getElementById('modalBody');
        
        modalTitle.textContent = `جزئیات ${symbol.toUpperCase()}`;
        modalBody.innerHTML = `
            <div class="coin-details">
                <div class="detail-section">
                    <h4>اطلاعات اصلی</h4>
                    <div class="detail-grid">
                        <div class="detail-item"><span>قیمت:</span> <span>$${this.formatPrice(data.price)}</span></div>
                        <div class="detail-item"><span>تغییر:</span> <span>${data.change}%</span></div>
                        <div class="detail-item"><span>حجم:</span> <span>${this.formatNumber(data.volume)}</span></div>
                        <div class="detail-item"><span>رتبه:</span> <span>${data.rank || '--'}</span></div>
                    </div>
                </div>
                ${data.signal ? `
                <div class="detail-section">
                    <h4>تحلیل</h4>
                    <div class="signal-info">
                        <span class="signal-badge ${data.signalClass}">${data.signalText}</span>
                    </div>
                </div>
                ` : ''}
            </div>
        `;
        
        this.showModal();
    }

    showAIDetails(symbol, analysis) {
        this.displayAIAnalysis(symbol, analysis);
    }

    // ==================== SYSTEM MANAGEMENT ====================

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
            const statusDot = document.getElementById('statusDot');
            const statusText = document.getElementById('statusText');
            statusDot.className = 'status-dot offline';
            statusText.textContent = 'خطا';
        }
    }

    loadDashboard() {
        document.getElementById('cacheCount').textContent = this.systemStats.cacheItems;
        document.getElementById('totalSymbols').textContent = this.top100Symbols.length;
        document.getElementById('scanCount').textContent = this.systemStats.todayScans;
        document.getElementById('aiAnalysisCount').textContent = this.systemStats.aiAnalyses;
        
        this.updateActivityList();
        this.updateAIStatusDisplay();
    }

    loadHealthStatus() {
        this.checkAPIStatus();
        this.checkAIStatus();
        this.updateLogsDisplay();
    }

    loadAISection() {
        this.updateAISection();
    }

    updateAISection() {
        const engineStatus = document.getElementById('aiEngineStatus');
        if (engineStatus) {
            engineStatus.innerHTML = `
                <div class="indicator">
                    <span class="indicator-label">موتور تکنیکال</span>
                    <span class="indicator-value">${this.aiStatus.engineReady ? 'فعال' : 'غیرفعال'}</span>
                </div>
                <div class="indicator">
                    <span class="indicator-label">تحلیل روند</span>
                    <span class="indicator-value">${this.aiStatus.initialized ? 'آماده' : 'در انتظار'}</span>
                </div>
                <div class="indicator">
                    <span class="indicator-label">داده‌های زنده</span>
                    <span class="indicator-value">${this.aiStatus.initialized ? 'متصل' : 'قطع'}</span>
                </div>
            `;
        }
    }

    updateAIStatusDisplay() {
        const aiStatusElement = document.getElementById('aiStatus');
        if (aiStatusElement) {
            aiStatusElement.innerHTML = `
                <div class="status-item">وضعیت: ${this.aiStatus.initialized ? 'فعال' : 'غیرفعال'}</div>
                <div class="status-item">تحلیل‌ها: ${this.aiAnalysisCount}</div>
                <div class="status-item">آخرین بروزرسانی: ${this.aiStatus.lastAnalysis ? new Date(this.aiStatus.lastAnalysis).toLocaleString('fa-IR') : '--'}</div>
            `;
        }
    }

    updateActivityList() {
        const activityList = document.getElementById('activityList');
        if (activityList) {
            const recentLogs = this.systemLogs.slice(-5).reverse();
            activityList.innerHTML = recentLogs.map(log => 
                `<div class="activity-item">${log.message}</div>`
            ).join('');
        }
    }

    updateLogsDisplay() {
        const logsContainer = document.getElementById('logsContainer');
        if (logsContainer) {
            logsContainer.innerHTML = this.systemLogs.slice(-20).reverse().map(log => `
                <div class="log-entry">
                    <span class="log-time">${new Date(log.timestamp).toLocaleTimeString('fa-IR')}</span>
                    <span class="log-level ${log.level}">${log.level.toUpperCase()}</span>
                    <span class="log-message">${log.message}</span>
                </div>
            `).join('');
        }
    }

    logSystem(message, level = 'info') {
        const logEntry = {
            timestamp: new Date(),
            message: message,
            level: level
        };
        
        this.systemLogs.push(logEntry);
        
        // حفظ حداکثر 100 لاگ
        if (this.systemLogs.length > 100) {
            this.systemLogs = this.systemLogs.slice(-100);
        }
        
        this.updateActivityList();
        this.updateLogsDisplay();
        
        // همچنین در کنسول مرورگر
        const consoleMethod = level === 'error' ? 'error' : level === 'warning' ? 'warn' : 'log';
        console[consoleMethod](`[VortexAI] ${message}`);
    }

    // ==================== OTHER METHODS ====================

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
        this.logSystem(`${count} ارز برتر انتخاب شدند`);
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

    clearResults() {
        const resultsGrid = document.getElementById('resultsGrid');
        const resultsCount = document.getElementById('resultsCount');
        
        if (resultsGrid) {
            resultsGrid.innerHTML = `
                <div class="empty-state">
                    <p>نتایج پاکسازی شد</p>
                </div>
            `;
        }
        
        if (resultsCount) {
            resultsCount.textContent = '0 مورد';
        }
        
        this.logSystem('نتایج اسکن پاکسازی شد', 'info');
    }

    clearLogs() {
        this.systemLogs = [];
        this.updateLogsDisplay();
        this.logSystem('لاگ‌های سیستم پاکسازی شدند', 'info');
    }

    exportData() {
        const data = {
            timestamp: new Date().toISOString(),
            scans: this.scanCount,
            aiAnalyses: this.aiAnalysisCount,
            symbols: this.top100Symbols.length,
            logs: this.systemLogs.slice(-10)
        };
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `vortexai-export-${new Date().toISOString().split('T')[0]}.json`;
        a.click();
        URL.revokeObjectURL(url);
        
        this.logSystem('داده‌های سیستم export شدند', 'success');
    }

    loadSettings() {
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
        this.logSystem('تنظیمات سیستم ذخیره شدند', 'success');
        alert('✅ تنظیمات ذخیره شد');
    }
}

// سیستم اسکن
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
            const batches = [];
            for (let i = 0; i < this.symbols.length; i += this.batchSize) {
                batches.push(this.symbols.slice(i, i + this.batchSize));
            }

            for (let i = 0; i < batches.length; i++) {
                if (this.isCancelled) break;

                const batch = batches[i];
                await this.processBatch(batch, i + 1, batches.length);
                
                if (i < batches.length - 1 && !this.isCancelled) {
                    await this.delay(500);
                }
            }

            if (!this.isCancelled) {
                this.onComplete(this.results);
            }

        } catch (error) {
            this.onError(error);
        }
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

        if (this.onProgress) {
            this.onProgress({
                completed: this.completed,
                total: this.symbols.length,
                batch: batchNumber,
                totalBatches: totalBatches,
                mode: this.mode
            });
        }
    }

    async scanSymbol(symbol) {
        try {
            let endpoint;
            if (this.mode === 'ai') {
                endpoint = `/api/ai/analyze`;
            } else {
                endpoint = `/api/scan/basic/${symbol}`;
            }

            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 20000);

            let response;
            if (this.mode === 'ai') {
                response = await fetch(endpoint, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        symbol: symbol,
                        analysis_type: 'comprehensive'
                    }),
                    signal: controller.signal
                });
            } else {
                response = await fetch(endpoint, {
                    signal: controller.signal
                });
            }
            
            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            
            return {
                symbol,
                success: true,
                data: this.mode === 'ai' ? data.data : data,
                mode: this.mode,
                timestamp: new Date().toISOString()
            };

        } catch (error) {
            console.error(`خطا در اسکن ${symbol}:`, error);
            return {
                symbol,
                success: false,
                error: error.message,
                mode: this.mode,
                timestamp: new Date().toISOString()
            };
        }
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    cancel() {
        this.isCancelled = true;
    }
}

// راه‌اندازی برنامه
const vortexApp = new VortexApp();
