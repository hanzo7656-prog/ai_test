// سیستم اصلی VortexAI - نسخه نهایی و یکپارچه
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
        
        this.logs = [];
        
        // bind methods
        this.boundHandleDocumentClick = this.handleDocumentClick.bind(this);
        this.boundHandleKeydown = this.handleKeydown.bind(this);
        this.boundHandleBeforeUnload = this.handleBeforeUnload.bind(this);
        
        this.init();
    }

    init() {
        console.log('🚀 Initializing VortexAI...');
        
        // بررسی وجود ماژول‌های ضروری
        this.checkRequiredModules();
        
        this.bindEvents();
        this.loadSettings();
        this.checkAPIStatus();
        this.showSection('scan');
        this.initConsole();
        this.startAutoHealthCheck();
        
        this.log('SUCCESS', 'سیستم VortexAI راه‌اندازی شد');
        this.uiManager.showNotification('VortexAI آماده است! 🚀', 'success');
    }

    checkRequiredModules() {
        const requiredModules = {
            'VortexUtils': typeof VortexUtils !== 'undefined',
            'UIManager': typeof UIManager !== 'undefined',
            'ScanSession': typeof ScanSession !== 'undefined',
            'AIClient': typeof AIClient !== 'undefined'
        };

        console.log('🔍 Checking required modules:', requiredModules);

        const missingModules = Object.entries(requiredModules)
            .filter(([_, available]) => !available)
            .map(([name]) => name);

        if (missingModules.length > 0) {
            console.error('❌ Missing modules:', missingModules);
            this.log('ERROR', `ماژول‌های ضروری بارگذاری نشدند: ${missingModules.join(', ')}`);
        } else {
            console.log('✅ All required modules loaded successfully');
        }
    }

    bindEvents() {
        try {
            // Navigation
            document.querySelectorAll('.nav-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const section = e.target.closest('.nav-btn').dataset.section;
                    this.showSection(section);
                    this.toggleMobileMenu(false);
                });
            });

            // Mobile Menu
            const mobileMenuBtn = document.getElementById('mobileMenuBtn');
            if (mobileMenuBtn) {
                mobileMenuBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.toggleMobileMenu();
                });
            }

            // Filter Menu
            const filterToggle = document.getElementById('filterToggle');
            if (filterToggle) {
                filterToggle.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.toggleFilterMenu();
                });
            }

            document.querySelectorAll('.filter-option').forEach(option => {
                option.addEventListener('click', (e) => {
                    const count = parseInt(e.target.dataset.count);
                    this.selectTopSymbols(count);
                    this.hideFilterMenu();
                });
            });

            // Scan Mode
            document.querySelectorAll('input[name="scanMode"]').forEach(radio => {
                radio.addEventListener('change', (e) => {
                    this.scanMode = e.target.value;
                    this.log('DEBUG', `حالت اسکن تغییر کرد به: ${this.scanMode}`);
                });
            });

            // Symbols Input
            const symbolsInput = document.getElementById('symbolsInput');
            if (symbolsInput) {
                symbolsInput.addEventListener('input', (e) => {
                    this.updateSelectedSymbols(e.target.value);
                });
            }

            // Scan Actions
            const startScan = document.getElementById('startScan');
            if (startScan) {
                startScan.addEventListener('click', () => {
                    this.startSmartScan();
                });
            }

            const clearResults = document.getElementById('clearResults');
            if (clearResults) {
                clearResults.addEventListener('click', () => {
                    this.clearResults();
                });
            }

            const exportResults = document.getElementById('exportResults');
            if (exportResults) {
                exportResults.addEventListener('click', () => {
                    this.exportResults();
                });
            }

            // Health Actions
            const refreshHealth = document.getElementById('refreshHealth');
            if (refreshHealth) {
                refreshHealth.addEventListener('click', () => {
                    this.loadHealthStatus();
                });
            }

            const testAPI = document.getElementById('testAPI');
            if (testAPI) {
                testAPI.addEventListener('click', () => {
                    this.testAPIEndpoints();
                });
            }

            // AI Actions
            const initAI = document.getElementById('initAI');
            if (initAI) {
                initAI.addEventListener('click', () => {
                    this.initAIEngine();
                });
            }

            const analyzeWithAI = document.getElementById('analyzeWithAI');
            if (analyzeWithAI) {
                analyzeWithAI.addEventListener('click', () => {
                    this.analyzeWithAI();
                });
            }

            // Log Actions
            const clearLogs = document.getElementById('clearLogs');
            if (clearLogs) {
                clearLogs.addEventListener('click', () => {
                    this.clearLogs();
                });
            }

            const exportLogs = document.getElementById('exportLogs');
            if (exportLogs) {
                exportLogs.addEventListener('click', () => {
                    this.exportLogs();
                });
            }

            // Loading Actions
            const cancelScan = document.getElementById('cancelScan');
            if (cancelScan) {
                cancelScan.addEventListener('click', () => {
                    this.cancelScan();
                });
            }

            const cancelLoading = document.getElementById('cancelLoading');
            if (cancelLoading) {
                cancelLoading.addEventListener('click', () => {
                    this.cancelScan();
                });
            }

            // Settings Actions
            const saveSettings = document.getElementById('saveSettings');
            if (saveSettings) {
                saveSettings.addEventListener('click', () => {
                    this.saveSettings();
                });
            }

            const clearCache = document.getElementById('clearCache');
            if (clearCache) {
                clearCache.addEventListener('click', () => {
                    this.clearCache();
                });
            }

            const resetSettings = document.getElementById('resetSettings');
            if (resetSettings) {
                resetSettings.addEventListener('click', () => {
                    this.resetSettings();
                });
            }

            const backupSettings = document.getElementById('backupSettings');
            if (backupSettings) {
                backupSettings.addEventListener('click', () => {
                    this.backupSettings();
                });
            }

            // Dashboard Actions
            const quickStats = document.getElementById('quickStats');
            if (quickStats) {
                quickStats.addEventListener('click', () => {
                    this.showQuickStats();
                });
            }

            const refreshDashboard = document.getElementById('refreshDashboard');
            if (refreshDashboard) {
                refreshDashboard.addEventListener('click', () => {
                    this.loadDashboard();
                });
            }

            // Global Event Listeners
            document.addEventListener('click', this.boundHandleDocumentClick);
            document.addEventListener('keydown', this.boundHandleKeydown);
            window.addEventListener('beforeunload', this.boundHandleBeforeUnload);

            this.log('SUCCESS', 'Event listeners initialized successfully');

        } catch (error) {
            console.error('Error in bindEvents:', error);
            this.log('ERROR', `خطا در راه‌اندازی event listeners: ${error.message}`);
        }
    }

    // ===== مدیریت ناوبری =====
    showSection(section) {
        try {
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

        } catch (error) {
            this.log('ERROR', `خطا در نمایش بخش ${section}: ${error.message}`);
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
            .filter(s => s.length > 0 && VortexUtils.isValidSymbol(s));
        
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
        console.log('🔍 Starting smart scan...');
        
        if (this.isScanning) {
            this.uiManager.showNotification('اسکن در حال انجام است', 'warning');
            return;
        }

        // دریافت ارزهای مورد نظر برای اسکن
        const symbolsToScan = this.selectedSymbols.length > 0 ? 
            this.selectedSymbols : this.top100Symbols.slice(0, this.batchSize);

        if (symbolsToScan.length === 0) {
            this.uiManager.showNotification('لطفاً حداقل یک ارز انتخاب کنید', 'error');
            return;
        }

        console.log(`🎯 Scan parameters:`, {
            symbols: symbolsToScan.length,
            mode: this.scanMode,
            batchSize: this.batchSize
        });

        this.isScanning = true;
        this.performanceStats.totalScans++;
        
        this.log('INFO', `شروع اسکن ${symbolsToScan.length} ارز در حالت ${this.scanMode}`);
        
        // نمایش لودینگ
        this.uiManager.showLoading();

        try {
            // بررسی وجود ScanSession
            if (typeof ScanSession === 'undefined') {
                throw new Error('سیستم اسکن بارگذاری نشده است');
            }

            // ایجاد session اسکن
            this.currentScan = new ScanSession({
                symbols: symbolsToScan,
                mode: this.scanMode,
                batchSize: this.batchSize,
                onProgress: (progress) => {
                    this.updateProgress(progress);
                },
                onComplete: (results) => {
                    this.onScanComplete(results);
                },
                onError: (error) => {
                    this.onScanError(error);
                }
            });

            // شروع اسکن
            await this.currentScan.start();

        } catch (error) {
            this.log('ERROR', `خطا در شروع اسکن: ${error.message}`);
            this.uiManager.showNotification('خطا در انجام اسکن', 'error');
            
            // پاکسازی در صورت خطا
            this.isScanning = false;
            this.uiManager.hideLoading();
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
        this.uiManager.showNotification(
            `✅ اسکن ${totalCount} ارز تکمیل شد (${successCount} موفق)`, 
            'success'
        );
        
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
        const aiRadio = document.querySelector('input[name="scanMode"][value="ai"]');
        if (aiRadio) aiRadio.checked = true;
        this.startSmartScan();
    }

    loadAIStatus() {
        const container = document.getElementById('aiStatusIndicators');
        if (!container) return;

        const status = this.aiClient.getStatus();
        this.uiManager.displayAIStatus(status);
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
        this.logs.push(logEntry);

        // نمایش در UI
        this.uiManager.displayLog(logEntry);

        // نمایش در کنسول مرورگر
        const styles = {
            'ERROR': 'color: #ff4757; font-weight: bold;',
            'WARN': 'color: #ff9f43; font-weight: bold;',
            'INFO': 'color: #0052ff; font-weight: bold;',
            'DEBUG': 'color: #64748b;',
            'SUCCESS': 'color: #00d9a6; font-weight: bold;'
        }[level];

        console.log(`%c[VortexAI] ${timestamp} ${level}: ${message}`, styles);
        if (data) console.log(data);
    }

    clearLogs() {
        this.logs = [];
        this.uiManager.clearLogs();
        this.log('INFO', 'همه لاگ‌ها پاکسازی شدند');
    }

    exportLogs() {
        if (this.logs.length === 0) {
            this.uiManager.showNotification('لاگی برای ذخیره وجود ندارد', 'warning');
            return;
        }

        const logText = this.logs.map(log => 
            `[${log.timestamp}] ${log.level}: ${log.message}`
        ).join('\n');

        VortexUtils.downloadFile('vortexai-logs.txt', logText);
        this.log('INFO', 'لاگ‌ها با موفقیت ذخیره شدند');
        this.uiManager.showNotification('لاگ‌ها ذخیره شدند', 'success');
    }

    // ===== سیستم سلامت و مانیتورینگ =====
    async loadHealthStatus() {
        try {
            this.log('DEBUG', 'دریافت وضعیت سلامت سیستم...');
            
            const response = await fetch('/api/status');
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const data = await response.json();
            
            this.uiManager.displayEndpointsHealth(data.endpoints_health || {});
            this.uiManager.displaySystemMetrics(data.system_metrics || {});
            this.uiManager.displayAIHealth(this.aiClient.getStatus());
            
            this.log('SUCCESS', 'وضعیت سلامت سیستم بروزرسانی شد');
        } catch (error) {
            this.log('ERROR', `خطا در دریافت وضعیت سلامت: ${error.message}`);
        }
    }

    async loadDashboard() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            // آپدیت آمار ساده
            const totalSymbols = document.getElementById('totalSymbols');
            const scanCount = document.getElementById('scanCount');
            const aiAnalysisCount = document.getElementById('aiAnalysisCount');
            
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

    // ===== سیستم تنظیمات =====
    loadSettings() {
        const settings = this.getStoredSettings();
        
        // بارگذاری تنظیمات در UI
        const batchSize = document.getElementById('batchSize');
        const cacheTTL = document.getElementById('cacheTTL');
        const resultsPerPage = document.getElementById('resultsPerPage');
        const aiPrecision = document.getElementById('aiPrecision');
        const autoLearning = document.getElementById('autoLearning');
        
        if (batchSize) batchSize.value = settings.batchSize;
        if (cacheTTL) cacheTTL.value = settings.cacheTTL;
        if (resultsPerPage) resultsPerPage.value = settings.resultsPerPage;
        if (aiPrecision) aiPrecision.value = settings.aiPrecision;
        if (autoLearning) autoLearning.checked = settings.autoLearning;

        this.uiManager.updateSystemInfo(this.performanceStats);
        
        this.log('DEBUG', 'تنظیمات از حافظه بارگذاری شد');
    }

    saveSettings() {
        const settings = {
            batchSize: document.getElementById('batchSize')?.value || '25',
            cacheTTL: document.getElementById('cacheTTL')?.value || '300',
            resultsPerPage: document.getElementById('resultsPerPage')?.value || '25',
            aiPrecision: document.getElementById('aiPrecision')?.value || 'medium',
            autoLearning: document.getElementById('autoLearning')?.checked || true,
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
            version: '3.0.0'
        };
        
        VortexUtils.downloadFile('vortexai-settings-backup.json', JSON.stringify(backupData, null, 2));
        this.log('INFO', 'پشتیبان تنظیمات ذخیره شد');
        this.uiManager.showNotification('💾 پشتیبان تنظیمات ذخیره شد', 'success');
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
        this.autoRefreshInterval = setInterval(() => {
            this.checkAPIStatus();
        }, 30000);
    }

    async checkAPIStatus() {
        try {
            const response = await fetch('/api/status');
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

    // ===== تست API endpoints =====
    async testAPIEndpoints() {
        this.log('INFO', '🧪 شروع تست API endpoints...');
        
        const testEndpoints = [
            { name: 'System Status', url: '/api/status' },
            { name: 'AI Status', url: '/api/ai/status' },
            { name: 'Raw BTC', url: '/api/raw/bitcoin' },
            { name: 'Processed BTC', url: '/api/processed/bitcoin' },
            { name: 'AI Technical', url: '/api/ai/analyze/bitcoin?analysis_type=technical' },
            { name: 'AI Sentiment', url: '/api/ai/analyze/bitcoin?analysis_type=sentiment' },
            { name: 'AI Prediction', url: '/api/ai/analyze/bitcoin?analysis_type=prediction' }
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
            
            await VortexUtils.delay(1000);
        }
        
        this.log('SUCCESS', '✅ تست API تکمیل شد');
        this.uiManager.showNotification('تست API انجام شد. نتیجه را در console ببینید.', 'info');
    }

    exportResults() {
        if (!this.currentScan || !this.currentScan.results || this.currentScan.results.length === 0) {
            this.uiManager.showNotification('هیچ نتیجه‌ای برای ذخیره وجود ندارد', 'warning');
            return;
        }

        const results = this.currentScan.results.filter(r => r.success);
        const csvContent = this.convertToCSV(results);
        VortexUtils.downloadFile('vortexai-results.csv', csvContent);
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

    // ===== کنسول توسعه =====
    initConsole() {
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
            analyze: (symbol) => this.analyzeSingleSymbol(symbol),
            utils: VortexUtils
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
        console.log('- vortex.utils - Utility functions');
    }

    async analyzeSingleSymbol(symbol) {
        this.log('INFO', `تحلیل تک ارز: ${symbol}`);
        this.uiManager.showNotification(`🧠 تحلیل ${symbol}...`, 'info');
        
        this.selectedSymbols = [symbol];
        this.scanMode = 'ai';
        const aiRadio = document.querySelector('input[name="scanMode"][value="ai"]');
        if (aiRadio) aiRadio.checked = true;
        
        this.startSmartScan();
    }

    // ===== Event Handlers =====
    handleDocumentClick(e) {
        // بستن منو فیلتر با کلیک خارج
        if (!e.target.closest('.currency-filter')) {
            this.hideFilterMenu();
        }

        // بستن منوی موبایل با کلیک خارج
        if (!e.target.closest('.nav-menu') && !e.target.closest('.mobile-menu-btn')) {
            this.toggleMobileMenu(false);
        }
    }

    handleKeydown(e) {
        this.handleKeyboard(e);
    }

    handleBeforeUnload(e) {
        if (this.isScanning) {
            e.preventDefault();
            e.returnValue = 'اسکن در حال انجام است. آیا مطمئنید که می‌خواهید صفحه را ترک کنید؟';
        }
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
                    const symbolsInput = document.getElementById('symbolsInput');
                    if (symbolsInput) symbolsInput.focus();
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

    // لیست کامل 100 ارز برتر
    top100Symbols = [
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
}

// راه‌اندازی برنامه
document.addEventListener('DOMContentLoaded', function() {
    console.log('📄 DOM Content Loaded - Initializing VortexAI...');
    try {
        window.vortexApp = new VortexApp();
        console.log('🎉 VortexAI initialized successfully!');
    } catch (error) {
        console.error('💥 Failed to initialize VortexAI:', error);
        alert('خطا در راه‌اندازی سیستم VortexAI. لطفاً صفحه را رفرش کنید.');
    }
});
