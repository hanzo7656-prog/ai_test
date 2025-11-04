// سیستم اصلی VortexAI
class VortexApp {
    constructor() {
        this.currentSection = 'scan';
        this.selectedSymbols = [];
        this.scanMode = 'basic';
        this.batchSize = 25;
        this.isScanning = false;
        this.currentScan = null;
        
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
        const topSymbols = this.getTopSymbols().slice(0, count);
        this.selectedSymbols = topSymbols;
        this.updateSymbolsInput();
    }

    getTopSymbols() {
        return [
            "bitcoin", "ethereum", "tether", "ripple", "binance-coin",
            "solana", "usd-coin", "cardano", "dogecoin", "polkadot",
            // ... بقیه ارزها
        ];
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
        countElement.textContent = `${this.selectedSymbols.length} ارز انتخاب شده`;
    }

    async startSmartScan() {
        if (this.isScanning) {
            alert('اسکن در حال انجام است!');
            return;
        }

        const symbolsToScan = this.selectedSymbols.length > 0 ? 
            this.selectedSymbols : this.getTopSymbols().slice(0, 100);

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
        document.getElementById('resultsGrid').innerHTML = `
            <div class="empty-state">
                <p>نتایج پاکسازی شد</p>
            </div>
        `;
        document.getElementById('resultsCount').textContent = '0 مورد';
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
        }
    }

    async loadDashboard() {
        // آپدیت آمار داشبورد
        document.getElementById('cacheCount').textContent = '0'; // از کش واقعی بگیر
        document.getElementById('totalSymbols').textContent = this.getTopSymbols().length;
        document.getElementById('apiStatus').textContent = 'متصل'; // از وضعیت واقعی بگیر
    }

    async loadHealthStatus() {
        try {
            const response = await fetch('/api/system/status');
            const data = await response.json();
            
            this.displayEndpointsHealth(data.endpoints_health);
            this.displaySystemMetrics(data.system_metrics);
            this.displayLogs(data);
            
        } catch (error) {
            console.error('خطا در دریافت وضعیت سلامت:', error);
        }
    }

    displayEndpointsHealth(endpoints) {
        const container = document.getElementById('endpointsList');
        if (!endpoints) return;

        let html = '';
        for (const [endpoint, info] of Object.entries(endpoints)) {
            const statusClass = info.status === 'success' ? 'status-success' : 'status-error';
            html += `
                <div class="endpoint-item">
                    <span class="endpoint-name">${endpoint}</span>
                    <span class="endpoint-status ${statusClass}">
                        ${info.status === 'success' ? 'فعال' : 'خطا'}
                    </span>
                </div>
            `;
        }
        container.innerHTML = html;
    }

    displaySystemMetrics(metrics) {
        const container = document.getElementById('systemMetrics');
        if (!metrics) return;

        container.innerHTML = `
            <div class="metric-item">مصرف CPU: ${metrics.cpu?.percent || 0}%</div>
            <div class="metric-item">مصرف RAM: ${metrics.memory?.percent || 0}%</div>
            <div class="metric-item">فضای دیسک: ${metrics.disk?.percent || 0}%</div>
        `;
    }

    displayLogs(data) {
        const container = document.getElementById('logsContainer');
        const timestamp = new Date().toLocaleString('fa-IR');
        
        let logs = `
            <div class="log-entry">
                <span class="log-time">${timestamp}</span>
                وضعیت سیستم: ${data.status}
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

        container.innerHTML = logs;
    }

    loadSettings() {
        // بارگذاری تنظیمات از localStorage
        const savedBatchSize = localStorage.getItem('vortex_batchSize') || '25';
        const savedCacheTTL = localStorage.getItem('vortex_cacheTTL') || '300';
        
        document.getElementById('batchSize').value = savedBatchSize;
        document.getElementById('cacheTTL').value = savedCacheTTL;
        
        this.batchSize = parseInt(savedBatchSize);
    }

    saveSettings() {
        const batchSize = document.getElementById('batchSize').value;
        const cacheTTL = document.getElementById('cacheTTL').value;
        
        localStorage.setItem('vortex_batchSize', batchSize);
        localStorage.setItem('vortex_cacheTTL', cacheTTL);
        
        this.batchSize = parseInt(batchSize);
        alert('تنظیمات ذخیره شد');
    }

    clearCache() {
        // پاکسازی کش
        localStorage.clear();
        alert('کش سیستم پاکسازی شد');
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
            }

            if (!this.isCancelled) {
                this.displayResults();
            }

        } catch (error) {
            console.error('خطا در اسکن:', error);
        } finally {
            vortexApp.hideLoading();
        }
    }

    async processBatch(batch, batchNumber, totalBatches) {
        const batchPromises = batch.map(symbol => this.scanSymbol(symbol));
        const batchResults = await Promise.all(batchPromises);
        
        this.results.push(...batchResults);
        this.completed += batch.length;

        this.updateLoadingUI(batch, batchNumber, totalBatches);
        this.displayPartialResults();
    }

    async scanSymbol(symbol) {
        try {
            const endpoint = this.mode === 'ai' ? 
                `/api/scan/ai/${symbol}` : `/api/scan/basic/${symbol}`;
            
            const response = await fetch(endpoint);
            if (!response.ok) throw new Error('خطا در دریافت داده');

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

    updateLoadingUI(currentBatch = [], batchNumber = 1, totalBatches = 1) {
        const total = this.symbols.length;
        const percent = Math.round((this.completed / total) * 100);
        const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
        const speed = elapsed > 0 ? Math.round((this.completed / elapsed) * 60) : 0;

        // آپدیت UI
        document.getElementById('progressText').textContent = `${this.completed}/${total}`;
        document.getElementById('progressPercent').textContent = `${percent}%`;
        document.getElementById('progressFill').style.width = `${percent}%`;
        document.getElementById('elapsedTime').textContent = this.formatTime(elapsed);
        document.getElementById('scanSpeed').textContent = `${speed}/دقیقه`;

        // نمایش ارزهای در حال اسکن
        const scanningList = document.getElementById('scanningList');
        if (currentBatch.length > 0) {
            scanningList.innerHTML = currentBatch
                .slice(0, 3)
                .map(symbol => `<span class="coin-tag">${symbol.toUpperCase()}</span>`)
                .join('');
        }

        // نمایش ارزهای تکمیل شده
        const completedList = document.getElementById('completedList');
        const completedSymbols = this.results
            .slice(-5)
            .filter(r => r.success)
            .map(r => r.symbol);
        
        completedList.innerHTML = completedSymbols
            .map(symbol => `<span class="coin-tag">${symbol.toUpperCase()}</span>`)
            .join('');
    }

    displayPartialResults() {
        const container = document.getElementById('resultsGrid');
        const countElement = document.getElementById('resultsCount');
        
        countElement.textContent = `${this.results.length} مورد`;
        
        if (this.results.length === 0) return;

        const html = this.results.map(result => this.createCoinCard(result)).join('');
        container.innerHTML = html;
    }

    displayResults() {
        this.displayPartialResults();
    }

    createCoinCard(result) {
        if (!result.success) {
            return `
                <div class="coin-card error">
                    <div class="coin-header">
                        <span class="coin-name">${result.symbol.toUpperCase()}</span>
                        <span class="coin-price">خطا</span>
                    </div>
                    <div class="coin-details">
                        <div class="detail-item">
                            <span class="detail-label">پیام:</span>
                            <span>${result.error}</span>
                        </div>
                    </div>
                </div>
            `;
        }

        const data = result.data.data || {};
        const displayData = data.display_data || {};
        const analysis = data.analysis || {};

        return `
            <div class="coin-card">
                <div class="coin-header">
                    <span class="coin-name">${result.symbol.toUpperCase()}</span>
                    <span class="coin-price">$${displayData.price || 0}</span>
                </div>
                <div class="coin-details">
                    <div class="detail-item">
                        <span class="detail-label">تغییر 24h:</span>
                        <span>${displayData.price_change_24h || 0}%</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">حجم:</span>
                        <span>${this.formatNumber(displayData.volume_24h || 0)}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">سیگنال:</span>
                        <span>${analysis.signal || 'HOLD'}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">اعتماد:</span>
                        <span>${Math.round((analysis.confidence || 0) * 100)}%</span>
                    </div>
                </div>
            </div>
        `;
    }

    formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    }

    formatNumber(num) {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        }
        if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toString();
    }

    cancel() {
        this.isCancelled = true;
    }
}

// راه‌اندازی برنامه
const vortexApp = new VortexApp();
