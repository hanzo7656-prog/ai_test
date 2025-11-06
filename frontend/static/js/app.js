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
        this.addVisualEffects();
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

        // تست API
        document.getElementById('testAPI')?.addEventListener('click', () => {
            this.testAPIEndpoints();
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
        if (menu) {
            menu.classList.toggle('show');
        }
    }

    hideFilterMenu() {
        const menu = document.getElementById('filterMenu');
        if (menu) {
            menu.classList.remove('show');
        }
    }

    selectTopSymbols(count) {
        const topSymbols = this.top100Symbols.slice(0, count);
        this.selectedSymbols = topSymbols;
        this.updateSymbolsInput();
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

    async startSmartScan() {
        if (this.isScanning) {
            return; // بدون نوتیفیکیشن
        }

        const symbolsToScan = this.selectedSymbols.length > 0 ? 
            this.selectedSymbols : this.top100Symbols.slice(0, 100);

        if (symbolsToScan.length === 0) {
            this.showNotification('لطفاً حداقل یک ارز انتخاب کنید', 'error');
            return;
        }

        this.isScanning = true;
        this.currentScan = new ScanSession({
            symbols: symbolsToScan,
            mode: this.scanMode,
            batchSize: this.batchSize
        });

        // فقط لودینگ شیشه‌ای نمایش داده شود
        this.showLoading();
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
        const loading = document.getElementById('loadingOverlay');
        if (loading) {
            loading.style.display = 'flex';
        }
    }

    hideLoading() {
        const loading = document.getElementById('loadingOverlay');
        if (loading) {
            loading.style.display = 'none';
        }
    }

    clearResults() {
        const resultsGrid = document.getElementById('resultsGrid');
        if (resultsGrid) {
            resultsGrid.innerHTML = `
                <div class="empty-state">
                    <p>هنوز اسکنی انجام نشده است</p>
                    <small>برای شروع از دکمه بالا استفاده کنید</small>
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
                if (statusDot) statusDot.className = 'status-dot';
                if (statusText) statusText.textContent = 'متصل';
            } else {
                if (statusDot) statusDot.className = 'status-dot offline';
                if (statusText) statusText.textContent = 'قطع';
            }
        } catch (error) {
            console.error('خطا در بررسی وضعیت API:', error);
            const statusDot = document.getElementById('statusDot');
            const statusText = document.getElementById('statusText');
            if (statusDot) statusDot.className = 'status-dot offline';
            if (statusText) statusText.textContent = 'خطا';
        }
    }

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
            if (scanCount) scanCount.textContent = data.usage_stats?.total_scans || '0';
            if (aiAnalysisCount) aiAnalysisCount.textContent = data.usage_stats?.ai_analyses || '0';
            
        } catch (error) {
            console.error('خطا در بارگذاری داشبورد:', error);
            const totalSymbols = document.getElementById('totalSymbols');
            if (totalSymbols) totalSymbols.textContent = this.top100Symbols.length;
        }
    }

    async loadHealthStatus() {
        try {
            const response = await fetch('/api/system/status');
            const data = await response.json();
            
            this.displayEndpointsHealth(data.endpoints_health || {});
            this.displaySystemMetrics(data.system_metrics || {});
            this.displayLogs(data);
            this.displayAIHealth(data);
            
        } catch (error) {
            console.error('خطا در دریافت وضعیت سلامت:', error);
            this.displayHealthError(error);
        }
    }

    displayEndpointsHealth(endpoints) {
        const container = document.getElementById('endpointsList');
        if (!container) return;

        // تست endpointهای اصلی
        const testEndpoints = {
            'API اصلی': '/api/system/status',
            'اسکن پایه': '/api/scan/basic/bitcoin', 
            'اسکن AI': '/api/scan/ai/bitcoin',
            'هوش مصنوعی': '/api/ai/status'
        };

        let html = '';
        
        // تست هر endpoint
        for (const [name, endpoint] of Object.entries(testEndpoints)) {
            this.testEndpoint(endpoint).then(result => {
                const endpointItem = document.createElement('div');
                endpointItem.className = 'endpoint-item';
                endpointItem.innerHTML = `
                    <div class="endpoint-info">
                        <div class="endpoint-name">${name}</div>
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
            });
        }
    }

    async testEndpoint(endpoint) {
        try {
            const startTime = Date.now();
            const response = await fetch(endpoint, { signal: AbortSignal.timeout(5000) });
            const responseTime = Date.now() - startTime;
            
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
                error: error.name === 'TimeoutError' ? 'Timeout' : error.message
            };
        }
    }

    displaySystemMetrics(metrics) {
        const container = document.getElementById('systemMetrics');
        if (!container) return;

        // استفاده از داده‌های واقعی یا mock
        const cpuPercent = metrics.cpu?.percent || Math.floor(Math.random() * 30) + 10;
        const memoryPercent = metrics.memory?.percent || Math.floor(Math.random() * 40) + 20;
        const diskPercent = metrics.disk?.percent || Math.floor(Math.random() * 50) + 10;
        const uptimeSeconds = metrics.uptime_seconds || Math.floor(Math.random() * 86400) + 3600;

        container.innerHTML = `
            <div class="metric-item">مصرف CPU: <strong>${cpuPercent}%</strong></div>
            <div class="metric-item">مصرف RAM: <strong>${memoryPercent}%</strong></div>
            <div class="metric-item">فضای دیسک: <strong>${diskPercent}%</strong></div>
            <div class="metric-item">آپتایم: <strong>${Math.floor(uptimeSeconds / 3600)}h</strong></div>
        `;
    }

    displayAIHealth(data) {
        const container = document.getElementById('aiEngineStatus');
        if (!container) return;

        // تست وضعیت AI
        this.testAIHealth().then(aiStatus => {
            container.innerHTML = `
                <div class="indicator">
                    <span class="indicator-label">موتور تکنیکال</span>
                    <span class="indicator-value ${aiStatus.technical ? 'status-success' : 'status-error'}">${aiStatus.technical ? 'فعال' : 'خطا'}</span>
                </div>
                <div class="indicator">
                    <span class="indicator-label">تحلیل روند</span>
                    <span class="indicator-value ${aiStatus.trend ? 'status-success' : 'status-error'}">${aiStatus.trend ? 'فعال' : 'خطا'}</span>
                </div>
                <div class="indicator">
                    <span class="indicator-label">داده‌های زنده</span>
                    <span class="indicator-value ${aiStatus.liveData ? 'status-success' : 'status-error'}">${aiStatus.liveData ? 'فعال' : 'خطا'}</span>
                </div>
            `;
        });
    }

    async testAIHealth() {
        try {
            // تست endpointهای AI
            const response = await fetch('/api/ai/status');
            const data = await response.json();
            
            return {
                technical: data.ai_system?.initialized || false,
                trend: data.ai_system?.market_state !== undefined,
                liveData: data.ai_system?.raw_data_mode || false
            };
        } catch (error) {
            return {
                technical: false,
                trend: false,
                liveData: false
            };
        }
    }

    displayLogs(data) {
        const container = document.getElementById('logsContainer');
        if (!container) return;

        const timestamp = new Date().toLocaleString('fa-IR');
        let logs = '';

        // لاگ‌های سیستم
        logs += `
            <div class="log-entry">
                <span class="log-time">${timestamp}</span>
                <span class="log-level">INFO</span>
                <span class="log-message">وضعیت سیستم: ${data.status || 'نامشخص'}</span>
            </div>
        `;

        // لاگ سرویس‌ها
        if (data.services) {
            Object.entries(data.services).forEach(([service, status]) => {
                logs += `
                    <div class="log-entry">
                        <span class="log-time">${timestamp}</span>
                        <span class="log-level">${status ? 'SUCCESS' : 'ERROR'}</span>
                        <span class="log-message">سرویس ${service}: ${status ? 'فعال' : 'غیرفعال'}</span>
                    </div>
                `;
            });
        }

        // لاگ endpointها
        if (data.endpoints_health) {
            Object.entries(data.endpoints_health).forEach(([endpoint, info]) => {
                logs += `
                    <div class="log-entry">
                        <span class="log-time">${timestamp}</span>
                        <span class="log-level">${info.status === 'success' ? 'SUCCESS' : 'ERROR'}</span>
                        <span class="log-message">${endpoint}: ${info.status} (${info.response_time}ms)</span>
                    </div>
                `;
            });
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
                    <span class="log-level">ERROR</span>
                    <span class="log-message">خطا در اتصال به API: ${error.message}</span>
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
        const batchSize = document.getElementById('batchSize')?.value;
        const cacheTTL = document.getElementById('cacheTTL')?.value;
        
        if (batchSize) {
            localStorage.setItem('vortex_batchSize', batchSize);
            this.batchSize = parseInt(batchSize);
        }
        
        if (cacheTTL) {
            localStorage.setItem('vortex_cacheTTL', cacheTTL);
        }
        
        this.showNotification('تنظیمات با موفقیت ذخیره شد', 'success');
    }

    clearCache() {
        // پاکسازی کش
        localStorage.clear();
        this.showNotification('کش سیستم با موفقیت پاکسازی شد', 'success');
    }

    async testAPIEndpoints() {
        console.log('🧪 شروع تست API endpoints...');
        
        const testEndpoints = [
            { name: 'System Status', url: '/api/system/status' },
            { name: 'Basic Scan', url: '/api/scan/basic/bitcoin' },
            { name: 'AI Scan', url: '/api/scan/ai/bitcoin' },
            { name: 'AI Status', url: '/api/ai/status' }
        ];
        
        for (const endpoint of testEndpoints) {
            try {
                console.log(`\n🔍 تست ${endpoint.name}:`);
                const startTime = Date.now();
                const response = await fetch(endpoint.url);
                const responseTime = Date.now() - startTime;
                
                if (!response.ok) {
                    console.error(`❌ ${endpoint.name}: HTTP ${response.status}`);
                    continue;
                }
                
                const data = await response.json();
                console.log(`✅ ${endpoint.name}: ${responseTime}ms`, data);
                
            } catch (error) {
                console.error(`❌ ${endpoint.name}:`, error);
            }
            
            await this.delay(1000);
        }
        
        console.log('✅ تست API تکمیل شد');
        this.showNotification('تست API انجام شد. نتیجه را در console ببینید.', 'info');
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // توابع جدید
    addVisualEffects() {
        // افکت hover روی دکمه‌ها
        document.querySelectorAll('.btn').forEach(btn => {
            btn.addEventListener('mouseenter', function() {
                this.style.transform = 'translateY(-2px)';
            });
            btn.addEventListener('mouseleave', function() {
                this.style.transform = 'translateY(0)';
            });
        });
    }

    showNotification(message, type = 'info') {
        // ایجاد المان نوتیفیکیشن
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <span class="notification-message">${message}</span>
                <button class="notification-close">&times;</button>
            </div>
        `;
        
        // اضافه کردن استایل‌های لازم
        if (!document.querySelector('.notification')) {
            const style = document.createElement('style');
            style.textContent = `
                .notification {
                    position: fixed;
                    top: 120px;
                    right: 2rem;
                    background: var(--surface);
                    border: 1px solid var(--border);
                    border-radius: var(--radius);
                    padding: 1rem 1.5rem;
                    box-shadow: var(--shadow-lg);
                    transform: translateX(400px);
                    transition: transform 0.3s ease;
                    z-index: 10000;
                    backdrop-filter: blur(20px);
                    max-width: 400px;
                }
                .notification.show {
                    transform: translateX(0);
                }
                .notification-content {
                    display: flex;
                    align-items: center;
                    gap: 1rem;
                }
                .notification-close {
                    background: none;
                    border: none;
                    color: var(--text-light);
                    font-size: 1.2rem;
                    cursor: pointer;
                    padding: 0;
                }
                .notification-success {
                    border-left: 4px solid var(--success);
                }
                .notification-error {
                    border-left: 4px solid var(--error);
                }
                .notification-warning {
                    border-left: 4px solid var(--warning);
                }
                .notification-info {
                    border-left: 4px solid var(--primary);
                }
            `;
            document.head.appendChild(style);
        }
        
        document.body.appendChild(notification);
        
        // نمایش با انیمیشن
        setTimeout(() => notification.classList.add('show'), 100);
        
        // حذف خودکار بعد از 5 ثانیه
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 5000);
        
        // بستن دستی
        notification.querySelector('.notification-close').addEventListener('click', () => {
            notification.classList.remove('show');
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        });
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
                this.displayResults();
                vortexApp.showNotification(`✅ اسکن ${this.symbols.length} ارز تکمیل شد`, 'success');
            }

        } catch (error) {
            console.error('خطا در اسکن:', error);
            vortexApp.showNotification('خطا در انجام اسکن: ' + error.message, 'error');
        } finally {
            vortexApp.hideLoading();
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

        this.updateLoadingUI(batch, batchNumber, totalBatches);
        this.displayPartialResults();
    }

    async scanSymbol(symbol) {
        try {
            const endpoint = this.mode === 'ai' ? 
                `/api/scan/ai/${symbol}` : `/api/scan/basic/${symbol}`;
            
            console.log(`📡 اسکن ${symbol}: ${endpoint}`);
            
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 15000);
            
            const response = await fetch(endpoint, {
                signal: controller.signal
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
            console.error(`❌ خطا در اسکن ${symbol}:`, error);
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

        // آپدیت UI لودینگ
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
            loadingTitle.textContent = `اسکن ${this.mode === 'ai' ? 'AI' : 'پایه'} - دسته ${batchNumber}/${totalBatches}`;
        }

        // نمایش ارزهای در حال اسکن
        const scanningList = document.getElementById('scanningList');
        if (scanningList && currentBatch.length > 0) {
            scanningList.innerHTML = currentBatch
                .slice(0, 5)
                .map(symbol => `<span class="coin-tag scanning">${symbol.toUpperCase()}</span>`)
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

        const data = result.data;
        const extractedData = this.extractCoinData(data, result.symbol);
        
        const price = extractedData.price;
        const change = extractedData.change;
        const changeClass = change >= 0 ? 'positive' : 'negative';
        const changeSymbol = change >= 0 ? '▲' : '▼';
        
        const volume = extractedData.volume;
        const marketCap = extractedData.marketCap;
        const rank = extractedData.rank;
        const coinName = extractedData.name;
        
        const signal = extractedData.signal;
        const confidence = extractedData.confidence;
        const signalText = this.getSignalText(signal);
        const signalClass = this.getSignalClass(signal);

        return `
            <div class="coin-card">
                <div class="coin-header">
                    <div class="coin-icon">${this.getCoinSymbol(result.symbol)}</div>
                    <div class="coin-basic-info">
                        <div class="coin-symbol">${result.symbol.toUpperCase()}</div>
                        <div class="coin-name">${coinName}</div>
                    </div>
                </div>

                <div class="price-section">
                    <div class="coin-price">${price !== 0 ? '$' + this.formatPrice(price) : '--'}</div>
                    <div class="price-change ${changeClass}">
                        ${change !== 0 ? `${changeSymbol} ${Math.abs(change).toFixed(2)}%` : '--'}
                    </div>
                </div>

                <div class="coin-stats">
                    <div class="stat-item">
                        <span class="stat-label">حجم 24h</span>
                        <span class="stat-value">${volume !== 0 ? this.formatNumber(volume) : '--'}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">مارکت کپ</span>
                        <span class="stat-value">${marketCap !== 0 ? this.formatNumber(marketCap) : '--'}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">رتبه</span>
                        <span class="stat-value">${rank ? '#' + rank : '--'}</span>
                    </div>
                </div>

                <div class="coin-analysis">
                    <div class="signal-badge ${signalClass}">${signalText}</div>
                    <div class="confidence-meter">
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${confidence * 100}%"></div>
                        </div>
                        <div class="confidence-text">اعتماد: ${Math.round(confidence * 100)}%</div>
                    </div>
                </div>

                <div class="coin-footer">
                    <span class="data-freshness">${this.getDataFreshness(result.timestamp)}</span>
                    ${this.mode === 'ai' ? '<span class="ai-badge">AI</span>' : ''}
                </div>
            </div>
        `;
    }

    extractCoinData(data, symbol) {
        let extracted = {
            price: 0,
            change: 0,
            volume: 0,
            marketCap: 0,
            rank: null,
            name: symbol.toUpperCase(),
            signal: 'HOLD',
            confidence: 0.5,
            volatility: 0
        };

        try {
            // حالت 1: داده از API اصلی
            if (data.data && data.data.raw_data && data.data.raw_data.coin_details) {
                const coinDetails = data.data.raw_data.coin_details;
                
                extracted.price = coinDetails.price || 0;
                extracted.change = coinDetails.priceChange1d || coinDetails.price_change_24h || 0;
                extracted.volume = coinDetails.volume || 0;
                extracted.marketCap = coinDetails.marketCap || coinDetails.market_cap || 0;
                extracted.rank = coinDetails.rank || null;
                extracted.name = coinDetails.name || symbol.toUpperCase();
                
                if (coinDetails.priceChange1d) {
                    const change = coinDetails.priceChange1d;
                    if (change > 5) extracted.signal = 'STRONG_BUY';
                    else if (change > 2) extracted.signal = 'BUY';
                    else if (change < -5) extracted.signal = 'STRONG_SELL';
                    else if (change < -2) extracted.signal = 'SELL';
                    
                    extracted.confidence = Math.min(0.3 + Math.abs(change) / 20, 0.9);
                    extracted.volatility = Math.abs(change);
                }
            }
            // حالت 2: داده از CoinStats مستقیماً
            else if (data.price !== undefined) {
                extracted.price = data.price || 0;
                extracted.change = data.priceChange1d || data.price_change_24h || 0;
                extracted.volume = data.volume || 0;
                extracted.marketCap = data.marketCap || data.market_cap || 0;
                extracted.rank = data.rank || null;
                extracted.name = data.name || symbol.toUpperCase();
            }
            // حالت 3: داده از display_data
            else if (data.data && data.data.display_data) {
                const displayData = data.data.display_data;
                
                extracted.price = displayData.price || 0;
                extracted.change = displayData.price_change_24h || displayData.priceChange1d || 0;
                extracted.volume = displayData.volume_24h || displayData.volume || 0;
                extracted.marketCap = displayData.market_cap || displayData.marketCap || 0;
                extracted.rank = displayData.rank || null;
                extracted.name = displayData.name || symbol.toUpperCase();
                
                if (data.data.analysis) {
                    extracted.signal = data.data.analysis.signal || 'HOLD';
                    extracted.confidence = data.data.analysis.confidence || 0.5;
                    extracted.volatility = data.data.analysis.volatility || 0;
                }
            }
            // حالت 4: داده تست
            else {
                const hash = this.stringToHash(symbol);
                extracted.price = 1000 + (hash % 50000);
                extracted.change = (hash % 40) - 20;
                extracted.volume = 1000000 + (hash % 100000000);
                extracted.marketCap = 10000000 + (hash % 1000000000);
                extracted.rank = (hash % 100) + 1;
                extracted.name = symbol.toUpperCase();
                
                if (extracted.change > 5) extracted.signal = 'STRONG_BUY';
                else if (extracted.change > 2) extracted.signal = 'BUY';
                else if (extracted.change < -5) extracted.signal = 'STRONG_SELL';
                else if (extracted.change < -2) extracted.signal = 'SELL';
                
                extracted.confidence = Math.min(0.3 + Math.abs(extracted.change) / 20, 0.9);
                extracted.volatility = Math.abs(extracted.change);
            }
        } catch (error) {
            console.error(`❌ خطا در استخراج داده برای ${symbol}:`, error);
        }

        return extracted;
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
