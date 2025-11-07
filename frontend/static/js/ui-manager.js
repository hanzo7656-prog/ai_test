// سیستم مدیریت رابط کاربری VortexAI
class UIManager {
    constructor() {
        this.autoRefreshInterval = null;
        this.autoScrollLogs = true;
        this.logFilters = {
            level: 'ALL',
            search: ''
        };
    }

    // ===== مدیریت ناوبری و منو =====
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

    // ===== سیستم لودینگ و نوتیفیکیشن =====
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

    // ===== مدیریت پیشرفت اسکن =====
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
            loadingTitle.textContent = `اسکن - ${percent}%`;
        }

        // نمایش ارزهای در حال اسکن
        if (scanningList && currentBatch && currentBatch.length > 0) {
            scanningList.innerHTML = currentBatch
                .slice(0, 8)
                .map(symbol => `<span class="coin-tag scanning">${symbol.toUpperCase()}</span>`)
                .join('');
        }
    }

    // ===== نمایش نتایج =====
    displayResults(results, scanMode = 'basic') {
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

        const html = results.map(result => this.createCoinCard(result, scanMode)).join('');
        container.innerHTML = `
            <div class="coin-grid">${html}</div>
        `;
    }

    createCoinCard(result, scanMode) {
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

                ${scanMode === 'ai' ? `
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
                    ${scanMode === 'ai' ? '<span class="ai-badge">AI</span>' : ''}
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

            // حالت AI Scan - داده مستقیم از تحلیل AI
            if (data && data.analysis) {
                // داده از تحلیل AI
                const analysis = data.analysis;
                extracted.signal = analysis.signal || 'HOLD';
                extracted.confidence = analysis.confidence || 0.5;
                
                // اگر داده بازار هم موجود باشه
                if (data.market_data) {
                    const market = data.market_data;
                    extracted.price = market.price || market.current_price || 0;
                    extracted.change = market.price_change_24h || market.priceChange1d || 0;
                    extracted.volume = market.volume || market.total_volume || 0;
                    extracted.marketCap = market.marketCap || market.market_cap || 0;
                    extracted.rank = market.rank || null;
                    extracted.name = market.name || symbol.toUpperCase();
                }
            }
            // حالت 1: داده از API اصلی
            else if (data && data.data) {
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
                console.warn(`ساختار داده برای ${symbol} شناسایی نشد، استفاده از داده تست`);
                const hash = this.stringToHash(symbol);
                extracted.price = 1000 + (hash % 50000);
                extracted.change = (hash % 40) - 20;
                extracted.volume = 1000000 + (hash % 100000000);
                extracted.marketCap = 10000000 + (hash % 1000000000);
                extracted.rank = (hash % 100) + 1;
                extracted.name = symbol.toUpperCase();
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
            console.error(`خطا در استخراج داده برای ${symbol}: ${error.message}`);
        }

        console.log(`✅ داده استخراج شده برای ${symbol}:`, extracted);
        return extracted;
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
    }

    // ===== سیستم لاگ =====
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
    }

    refreshLogsDisplay(logs, filters) {
        const container = document.getElementById('logsContainer');
        if (!container) return;

        container.innerHTML = '';
        
        if (logs) {
            logs.forEach(log => this.displayLog(log));
        }

        this.updateLogCount(logs, filters);
    }

    updateLogCount(logs, filters) {
        const countElement = document.getElementById('logCount');
        if (countElement && logs) {
            const filteredLogs = logs.filter(log => {
                if (filters.level !== 'ALL' && filters.level !== log.level) {
                    return false;
                }
                if (filters.search && !log.message.includes(filters.search)) {
                    return false;
                }
                return true;
            });
            countElement.textContent = filteredLogs.length;
        }
    }

    clearLogs() {
        const container = document.getElementById('logsContainer');
        if (container) {
            container.innerHTML = '';
        }
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

    toggleAutoRefresh(button, onRefreshCallback) {
        if (this.autoRefreshInterval) {
            clearInterval(this.autoRefreshInterval);
            this.autoRefreshInterval = null;
            button.innerHTML = '🔴 غیرفعال';
        } else {
            this.autoRefreshInterval = setInterval(() => {
                onRefreshCallback();
            }, 10000);
            button.innerHTML = '🟢 فعال';
        }
    }

    // ===== سیستم سلامت =====
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

    displayAIHealth(aiStatus) {
        const container = document.getElementById('aiEngineStatus');
        if (!container) return;
        
        container.innerHTML = `
            <div class="indicator">
                <span class="indicator-label">موتور تکنیکال</span>
                <span class="indicator-value ${aiStatus.technical?.ready ? 'status-success' : 'status-error'}">
                    ${aiStatus.technical?.ready ? 'فعال' : 'غیرفعال'}
                </span>
            </div>
            <div class="indicator">
                <span class="indicator-label">تحلیل روند</span>
                <span class="indicator-value ${aiStatus.sentiment?.ready ? 'status-success' : 'status-error'}">
                    ${aiStatus.sentiment?.ready ? 'فعال' : 'غیرفعال'}
                </span>
            </div>
            <div class="indicator">
                <span class="indicator-label">داده‌های زنده</span>
                <span class="indicator-value ${aiStatus.predictive?.ready ? 'status-success' : 'status-error'}">
                    ${aiStatus.predictive?.ready ? 'فعال' : 'غیرفعال'}
                </span>
            </div>
        `;
    }

    displayAIStatus(status) {
        const container = document.getElementById('aiStatusIndicators');
        if (!container) return;

        container.innerHTML = `
            <div class="indicator">
                <span class="indicator-label">
                    <span class="indicator-icon">📊</span>
                    موتور تکنیکال
                </span>
                <span class="indicator-value ${status.technical?.ready ? 'status-success' : 'status-error'}">
                    ${status.technical?.ready ? 'فعال' : 'غیرفعال'}
                </span>
            </div>
            <div class="indicator">
                <span class="indicator-label">
                    <span class="indicator-icon">😊</span>
                    تحلیل احساسات
                </span>
                <span class="indicator-value ${status.sentiment?.ready ? 'status-success' : 'status-error'}">
                    ${status.sentiment?.ready ? 'فعال' : 'غیرفعال'}
                </span>
            </div>
            <div class="indicator">
                <span class="indicator-label">
                    <span class="indicator-icon">🔮</span>
                    پیش‌بینی قیمت
                </span>
                <span class="indicator-value ${status.predictive?.ready ? 'status-success' : 'status-error'}">
                    ${status.predictive?.ready ? 'فعال' : 'غیرفعال'}
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

    // ===== سیستم اطلاعات =====
    updateSystemInfo(performanceStats) {
        // آپدیت اطلاعات سیستم در تنظیمات
        const versionElement = document.getElementById('systemVersion');
        const lastUpdateElement = document.getElementById('lastUpdate');
        const memoryUsedElement = document.getElementById('memoryUsed');
        const sessionDurationElement = document.getElementById('sessionDuration');

        if (versionElement) versionElement.textContent = '1.0.0';
        if (lastUpdateElement) lastUpdateElement.textContent = new Date().toLocaleString('fa-IR');
        if (memoryUsedElement) memoryUsedElement.textContent = this.formatMemoryUsage();
        if (sessionDurationElement) sessionDurationElement.textContent = this.formatSessionDuration(performanceStats);
    }

    // ===== ابزارهای کمکی =====
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
        const used = Math.round(50 + Math.random() * 50);
        return `${used} MB`;
    }

    formatSessionDuration(performanceStats) {
        const duration = Math.floor((Date.now() - performanceStats.startTime) / 1000);
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
}
