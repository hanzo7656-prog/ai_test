// سیستم اصلی VortexAI - نسخه ماژولار
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

        // سیستم‌های خارجی
        this.aiClient = new AIClient();
        this.uiManager = new UIManager();
        this.smartLoading = new SmartLoading();
        
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
        
        this.logs = [];
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

        // تنظیمات
        document.getElementById('saveSettings').addEventListener('click', () => {
            this.saveSettings();
        });

        document.getElementById('clearCache').addEventListener('click', () => {
            this.clearCache();
        });

        // AI
        document.getElementById('initAI').addEventListener('click', () => {
            this.initAIEngine();
        });

        document.getElementById('analyzeWithAI').addEventListener('click', () => {
            this.analyzeWithAI();
        });

        // سیستم لاگ
        document.getElementById('clearLogs').addEventListener('click', () => {
            this.clearLogs();
        });

        document.getElementById('exportLogs').addEventListener('click', () => {
            this.exportLogs();
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
        this.uiManager.toggleMobileMenu(force);
    }

    toggleFilterMenu() {
        this.uiManager.toggleFilterMenu();
    }

    hideFilterMenu() {
        this.uiManager.hideFilterMenu();
    }

    // ===== مدیریت ارزها =====
    selectTopSymbols(count) {
        const topSymbols = this.top100Symbols.slice(0, count);
        this.selectedSymbols = topSymbols;
        this.updateSymbolsInput();
        this.log('INFO', `${count} ارز برتر انتخاب شد`);
        this.uiManager.showNotification(`✅ ${count} ارز برتر انتخاب شد`, 'success');
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
            this.uiManager.showNotification('اسکن در حال انجام است', 'warning');
            return;
        }

        const symbolsToScan = this.selectedSymbols.length > 0 ? 
            this.selectedSymbols : this.top100Symbols.slice(0, this.batchSize);

        if (symbolsToScan.length === 0) {
            this.uiManager.showNotification('لطفاً حداقل یک ارز انتخاب کنید', 'error');
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
        this.uiManager.showLoading();
        
        try {
            await this.currentScan.start();
        } catch (error) {
            this.log('ERROR', `خطا در اسکن: ${error.message}`);
            this.uiManager.showNotification('خطا در انجام اسکن', 'error');
        }
    }

    updateProgress(progress) {
        this.uiManager.updateProgress(progress);
    }

    onScanComplete(results) {
        this.isScanning = false;
        this.uiManager.hideLoading();
        
        const successCount = results.filter(r => r.success).length;
        const totalCount = results.length;
        
        this.performanceStats.successfulScans += successCount;
        this.performanceStats.failedScans += (totalCount - successCount);
        
        // نمایش نتایج
        this.uiManager.displayResults(results, this.scanMode);
        
        this.log('SUCCESS', `اسکن تکمیل شد: ${successCount}/${totalCount} موفق`);
        this.uiManager.showNotification(`✅ اسکن ${totalCount} ارز تکمیل شد (${successCount} موفق)`, 'success');
        
        this.updatePerformanceStats();
    }

    onScanError(error) {
        this.isScanning = false;
        this.uiManager.hideLoading();
        
        this.performanceStats.failedScans++;
        this.log('ERROR', `خطا در اسکن: ${error.message}`);
        this.uiManager.showNotification('خطا در انجام اسکن', 'error');
        
        this.updatePerformanceStats();
    }

    // ===== هوش مصنوعی =====
    async initAIEngine() {
        this.log('INFO', '🚀 راه‌اندازی موتور AI...');
        this.uiManager.showLoading();
        
        try {
            const success = await this.aiClient.initialize();
            
            if (success) {
                this.log('SUCCESS', '✅ موتور AI با موفقیت راه‌اندازی شد');
                this.uiManager.showNotification('🤖 موتور AI فعال شد', 'success');
                this.loadAIStatus();
            } else {
                throw new Error('راه‌اندازی AI ناموفق بود');
            }
        } catch (error) {
            this.log('ERROR', `خطا در راه‌اندازی AI: ${error.message}`);
            this.uiManager.showNotification('خطا در راه‌اندازی AI', 'error');
        } finally {
            this.uiManager.hideLoading();
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
        this.uiManager.showNotification(`🧠 تحلیل ${symbol}...`, 'info');
        
        this.selectedSymbols = [symbol];
        this.scanMode = 'ai';
        document.querySelector('input[name="scanMode"][value="ai"]').checked = true;
        
        this.startSmartScan();
    }

    loadAIStatus() {
        const container = document.getElementById('aiStatusIndicators');
        if (!container) return;

        const status = this.aiClient.getStatus();
        this.uiManager.displayAIStatus(status);
    }

    cancelScan() {
        if (this.currentScan) {
            this.currentScan.cancel();
            this.log('INFO', 'اسکن توسط کاربر لغو شد');
        }
        this.isScanning = false;
        this.uiManager.hideLoading();
        this.uiManager.showNotification('اسکن لغو شد', 'warning');
    }

    clearResults() {
        this.uiManager.clearResults();
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
        this.uiManager.displayLog(logEntry);

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
        this.uiManager.updateLogCount();
    }

    setLogFilter(type, value) {
        this.logFilters[type] = value;
        this.uiManager.setLogFilter(type, value);
    }

    refreshLogsDisplay() {
        this.uiManager.refreshLogsDisplay(this.logs, this.logFilters);
    }

    updateLogCount() {
        this.uiManager.updateLogCount(this.logs, this.logFilters);
    }

    clearLogs() {
        this.logs = [];
        this.uiManager.clearLogs();
        this.log('INFO', 'همه لاگ‌ها پاکسازی شدند');
    }

    exportLogs() {
        if (!this.logs || this.logs.length === 0) {
            this.uiManager.showNotification('لاگی برای ذخیره وجود ندارد', 'warning');
            return;
        }

        const logText = this.logs.map(log => 
            `[${log.timestamp}] ${log.level}: ${log.message}`
        ).join('\n');

        this.downloadFile('vortexai-logs.txt', logText);
        this.log('INFO', 'لاگ‌ها با موفقیت ذخیره شدند');
        this.uiManager.showNotification('لاگ‌ها ذخیره شدند', 'success');
    }

    scrollLogsToBottom() {
        this.uiManager.scrollLogsToBottom();
    }

    scrollLogsToTop() {
        this.uiManager.scrollLogsToTop();
    }

    toggleAutoRefresh(button) {
        this.uiManager.toggleAutoRefresh(button, this.loadHealthStatus.bind(this));
    }

    // ===== سیستم سلامت و مانیتورینگ =====
    async loadHealthStatus() {
        try {
            this.log('DEBUG', 'دریافت وضعیت سلامت سیستم...');
            
            const response = await fetch('/api/system/status');
            const data = await response.json();
            
            this.uiManager.displayEndpointsHealth(data.endpoints_health || {});
            this.uiManager.displaySystemMetrics(data.system_metrics || {});
            this.uiManager.displayAIHealth(this.aiClient.getStatus());
            
            this.log('SUCCESS', 'وضعیت سلامت سیستم بروزرسانی شد');
        } catch (error) {
            this.log('ERROR', `خطا در دریافت وضعیت سلامت: ${error.message}`);
            this.uiManager.displayHealthError(error);
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
        this.uiManager.showNotification('✅ تنظیمات ذخیره شد', 'success');
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
        this.uiManager.showNotification('🗑️ کش سیستم پاکسازی شد', 'success');
    }

    resetSettings() {
        localStorage.removeItem('vortex_settings');
        this.loadSettings();
        this.log('INFO', 'تنظیمات به حالت پیش‌فرض بازگردانی شد');
        this.uiManager.showNotification('🔄 تنظیمات بازنشانی شد', 'success');
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
        this.uiManager.showNotification('💾 پشتیبان تنظیمات ذخیره شد', 'success');
    }

    updateSystemInfo() {
        this.uiManager.updateSystemInfo(this.performanceStats);
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
• وضعیت AI: ${this.aiClient.isInitialized ? 'فعال' : 'غیرفعال'}
        `.trim();

        this.log('INFO', 'آمار سریع سیستم:\n' + stats);
        this.uiManager.showNotification('📊 آمار سیستم نمایش داده شد', 'info');
    }

    // ===== ابزارهای کمکی =====
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
            settings: () => this.getStoredSettings(),
            scan: (symbols = ['bitcoin']) => {
                this.selectedSymbols = symbols;
                this.startSmartScan();
            },
            analyze: (symbol) => this.analyzeSingleSymbol(symbol)
        };

        console.log('🚀 VortexAI Console Activated!');
        console.log('Available commands:');
        console.log('- vortex.test() - Test API endpoints');
        console.log('- vortex.scan([symbols]) - Start scan');
        console.log('- vortex.analyze(symbol) - Analyze single symbol');
        console.log('- vortex.logs - View logs');
        console.log('- vortex.stats - View performance stats');
        console.log('- vortex.settings - View settings');
        console.log('- vortex.clear() - Clear logs');
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
            { name: 'Raw Data', url: '/api/raw/bitcoin' },
            { name: 'Processed Data', url: '/api/processed/bitcoin' },
            { name: 'AI Technical', url: '/api/ai/analyze/bitcoin?analysis_type=technical' },
            { name: 'AI Prediction', url: '/api/ai/analyze/bitcoin?analysis_type=prediction' },
            { name: 'System Status', url: '/api/status' },
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
        this.uiManager.showNotification('تست API انجام شد. نتیجه را در console ببینید.', 'info');
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    exportResults() {
        if (!this.currentScan || !this.currentScan.results || this.currentScan.results.length === 0) {
            this.uiManager.showNotification('هیچ نتیجه‌ای برای ذخیره وجود ندارد', 'warning');
            return;
        }

        const results = this.currentScan.results.filter(r => r.success);
        const csvContent = this.convertToCSV(results);
        this.downloadFile('vortexai-results.csv', csvContent);
        this.log('INFO', 'نتایج اسکن ذخیره شد');
        this.uiManager.showNotification('📥 نتایج ذخیره شد', 'success');
    }

    convertToCSV(results) {
        const headers = ['Symbol', 'Name', 'Price', 'Change%', 'Volume', 'MarketCap', 'Rank', 'Signal', 'Confidence'];
        const rows = results.map(result => {
            const data = this.uiManager.extractCoinData(result.data, result.symbol);
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
}

// راه‌اندازی برنامه
document.addEventListener('DOMContentLoaded', function() {
    window.vortexApp = new VortexApp();
});
