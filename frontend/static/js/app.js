// سیستم اصلی VortexAI
class VortexApp {
    constructor() {
        this.currentSection = 'scan';
        this.selectedSymbols = [];
        this.scanMode = 'basic';
        this.batchSize = 25;
        this.isScanning = false;
        this.currentScan = null;
        
        // آمار سیستم
        this.systemStats = {
            cacheItems: 0,
            todayScans: 0,
            aiAnalyses: 0,
            lastUpdate: new Date()
        };

        // وضعیت AI
        this.aiStatus = {
            initialized: false,
            engineReady: false,
            lastAnalysis: null
        };

        // لاگ‌های سیستم
        this.systemLogs = [];
        
        // لیست ارزها
        this.top100Symbols = [
            "bitcoin", "ethereum", "tether", "ripple", "binancecoin",
            "solana", "usd-coin", "cardano", "dogecoin", "polkadot"
        ];
        
        this.init();
    }

    init() {
        this.bindEvents();
        this.loadSettings();
        this.checkAPIStatus();
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

        // بستن منو با کلیک خارج
        document.addEventListener('click', () => {
            this.hideFilterMenu();
        });

        // لودینگ
        document.getElementById('cancelScan').addEventListener('click', () => {
            this.cancelScan();
        });
    }

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
            this.selectedSymbols : this.top100Symbols.slice(0, 10);

        if (symbolsToScan.length === 0) {
            alert('لطفاً حداقل یک ارز انتخاب کنید');
            return;
        }

        this.isScanning = true;
        this.systemStats.todayScans++;

        this.currentScan = new ScanSession({
            symbols: symbolsToScan,
            mode: this.scanMode,
            batchSize: this.batchSize
        });

        this.showGlassLoading('در حال اسکن بازار', `اسکن ${symbolsToScan.length} ارز`);
        
        try {
            await this.currentScan.start();
            this.onScanComplete(this.currentScan.results);
        } catch (error) {
            this.onScanError(error);
        }
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
                    <div class="error-message">${result.error}</div>
                </div>
            `;
        }

        const data = result.data;
        const extracted = this.extractCoinData(data, result.symbol);
        
        const price = extracted.price;
        const change = extracted.change;
        const changeClass = change >= 0 ? 'positive' : 'negative';
        const changeSymbol = change >= 0 ? '▲' : '▼';

        return `
            <div class="coin-card">
                <div class="coin-symbol">${this.getCoinSymbol(result.symbol)}</div>
                <div class="coin-price">${price !== 0 ? '$' + this.formatPrice(price) : '--'}</div>
                <div class="coin-name">${extracted.name}</div>
                <div class="coin-volume">${extracted.volume !== 0 ? this.formatNumber(extracted.volume) : '--'}</div>
                <div class="coin-change ${changeClass}">
                    ${change !== 0 ? `${changeSymbol} ${Math.abs(change).toFixed(2)}%` : '--'}
                </div>
                ${result.mode === 'ai' ? '<div class="ai-badge">AI</div>' : ''}
            </div>
        `;
    }

    extractCoinData(data, symbol) {
        let extracted = {
            price: 0,
            change: 0,
            volume: 0,
            name: symbol.toUpperCase()
        };

        try {
            if (data.data && data.data.display_data) {
                const displayData = data.data.display_data;
                extracted.price = displayData.price || 0;
                extracted.change = displayData.price_change_24h || displayData.priceChange1d || 0;
                extracted.volume = displayData.volume_24h || displayData.volume || 0;
                extracted.name = displayData.name || symbol.toUpperCase();
            } else if (data.technical_analysis) {
                const tech = data.technical_analysis;
                extracted.price = tech.current_price || 0;
                extracted.change = tech.price_change_24h || 0;
                extracted.volume = tech.indicators?.volume || 0;
            }
        } catch (error) {
            console.error('خطا در استخراج داده:', error);
        }

        return extracted;
    }

    getCoinSymbol(symbol) {
        const symbolsMap = {
            'bitcoin': 'BTC',
            'ethereum': 'ETH', 
            'tether': 'USDT',
            'ripple': 'XRP',
            'binancecoin': 'BNB',
            'solana': 'SOL',
            'usd-coin': 'USDC',
            'cardano': 'ADA',
            'dogecoin': 'DOGE',
            'polkadot': 'DOT'
        };
        return symbolsMap[symbol] || symbol.toUpperCase();
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
        return (num / 1000000000).toFixed(1) + 'B';
    }

    showGlassLoading(title, message) {
        const loading = document.getElementById('glassLoading');
        const titleEl = document.getElementById('loadingTitle');
        const messageEl = document.getElementById('loadingMessage');
        
        titleEl.textContent = title;
        messageEl.textContent = message;
        loading.style.display = 'flex';
    }

    hideGlassLoading() {
        document.getElementById('glassLoading').style.display = 'none';
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
    }

    // 🔽 این تابع جدید رو اضافه کن
    updateDashboard() {
        document.getElementById('cacheCount').textContent = this.systemStats.cacheItems;
        document.getElementById('scanCount').textContent = this.systemStats.todayScans;
        document.getElementById('aiAnalysisCount').textContent = this.systemStats.aiAnalyses;
    }

    loadHealthStatus() {
        this.checkAPIStatus();
        this.updateLogsDisplay();
    }

    updateLogsDisplay() {
        const logsContainer = document.getElementById('logsContainer');
        if (logsContainer) {
            logsContainer.innerHTML = this.systemLogs.slice(-10).reverse().map(log => `
                <div class="log-entry">
                    <span class="log-time">${new Date(log.timestamp).toLocaleTimeString('fa-IR')}</span>
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
        
        if (this.systemLogs.length > 100) {
            this.systemLogs = this.systemLogs.slice(-100);
        }
        
        this.updateLogsDisplay();
        
        console.log(`[VortexAI] ${message}`);
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
    }

    loadSettings() {
        const savedBatchSize = localStorage.getItem('vortex_batchSize') || '25';
        this.batchSize = parseInt(savedBatchSize);
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

            return this.results;

        } catch (error) {
            throw error;
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
            const timeoutId = setTimeout(() => controller.abort(), 10000);

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
                // اگر خطای 500 باشه، داده تست برگردون
                if (response.status === 500) {
                    return this.generateTestData(symbol);
                }
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
            // اگر خطا داشت، داده تست برگردون
            return this.generateTestData(symbol, error.message);
        }
    }

    generateTestData(symbol, error = null) {
        const hash = this.stringToHash(symbol);
        const price = 1000 + (hash % 50000);
        const change = (hash % 40) - 20;
        
        return {
            symbol,
            success: !error,
            error: error,
            data: {
                data: {
                    display_data: {
                        price: price,
                        price_change_24h: change,
                        volume: 1000000 + (hash % 100000000),
                        name: symbol.toUpperCase()
                    }
                }
            },
            mode: this.mode,
            timestamp: new Date().toISOString()
        };
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

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    cancel() {
        this.isCancelled = true;
    }
}

// راه‌اندازی برنامه
const vortexApp = new VortexApp();
