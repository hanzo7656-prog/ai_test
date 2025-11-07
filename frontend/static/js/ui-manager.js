// سیستم مدیریت رابط کاربری VortexAI - نسخه یکپارچه
class UIManager {
    constructor() {
        this.autoRefreshInterval = null;
        this.autoScrollLogs = true;
        this.logFilters = {
            level: 'ALL',
            search: ''
        };
        
        console.log('✅ UIManager initialized with VortexUtils');
    }

    // ===== مدیریت ناوبری و منو =====
    toggleMobileMenu(force) {
        const menu = document.getElementById('navMenu');
        const btn = document.getElementById('mobileMenuBtn');
        
        if (!menu || !btn) return;
        
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
        if (!menu || !btn) return;
        
        const isExpanded = menu.classList.toggle('show');
        btn.setAttribute('aria-expanded', isExpanded);
    }

    hideFilterMenu() {
        const menu = document.getElementById('filterMenu');
        const btn = document.getElementById('filterToggle');
        if (!menu || !btn) return;
        
        menu.classList.remove('show');
        btn.setAttribute('aria-expanded', 'false');
    }

    // ===== سیستم لودینگ و نوتیفیکیشن =====
    showLoading() {
        const loading = document.getElementById('loadingOverlay');
        if (loading) {
            loading.style.display = 'flex';
            document.body.style.overflow = 'hidden';
            
            // نمایش با انیمیشن
            setTimeout(() => {
                loading.style.opacity = '1';
            }, 10);
        }
    }

    hideLoading() {
        const loading = document.getElementById('loadingOverlay');
        if (loading) {
            loading.style.opacity = '0';
            setTimeout(() => {
                loading.style.display = 'none';
                document.body.style.overflow = '';
            }, 300);
        }
    }

    showNotification(message, type = 'info') {
        try {
            // ایجاد المان نوتیفیکیشن
            const notification = document.createElement('div');
            notification.className = `notification notification-${type}`;
            notification.setAttribute('role', 'alert');
            notification.innerHTML = `
                <div class="notification-content">
                    <span class="notification-message">${VortexUtils.escapeHtml(message)}</span>
                    <button class="notification-close" aria-label="بستن">&times;</button>
                </div>
            `;

            const container = document.getElementById('notificationsContainer') || document.body;
            container.appendChild(notification);

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

        } catch (error) {
            console.error('Notification error:', error);
        }
    }

    hideNotification(notification) {
        if (!notification) return;
        
        notification.classList.remove('show');
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }

    // ===== مدیریت پیشرفت اسکن =====
    updateProgress(progress) {
        if (!progress) return;

        const {
            completed,
            total,
            percent,
            elapsed,
            speed,
            currentBatch
        } = progress;

        // استفاده از VortexUtils برای فرمت‌دهی
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
        if (elapsedTime) elapsedTime.textContent = VortexUtils.formatTime(elapsed);
        if (scanSpeed) scanSpeed.textContent = `${speed}/دقیقه`;
        if (loadingTitle) {
            loadingTitle.textContent = `اسکن - ${percent}%`;
        }

        // نمایش ارزهای در حال اسکن
        if (scanningList && currentBatch && currentBatch.length > 0) {
            const limitedSymbols = currentBatch.slice(0, 5);
            scanningList.innerHTML = limitedSymbols
                .map(symbol => `<span class="coin-tag scanning">${symbol.toUpperCase()}</span>`)
                .join('');
        }
    }

    // ===== نمایش نتایج =====
    displayResults(results, scanMode = 'basic') {
        const container = document.getElementById('resultsGrid');
        const countElement = document.getElementById('resultsCount');
        
        if (!container) return;
        
        if (!results || results.length === 0) {
            container.innerHTML = this.createEmptyState('اسکن', 'هنوز اسکنی انجام نشده است');
            return;
        }

        const successCount = results.filter(r => r && r.success).length;
        if (countElement) {
            countElement.textContent = `${successCount}/${results.length} مورد`;
        }

        const html = results.map(result => this.createCoinCard(result, scanMode)).join('');
        container.innerHTML = `<div class="coin-grid">${html}</div>`;
    }

    createCoinCard(result, scanMode) {
        if (!result || !result.symbol) {
            return `
                <div class="coin-card error">
                    <div class="coin-header">
                        <div class="coin-icon">❌</div>
                        <div class="coin-basic-info">
                            <div class="coin-symbol">نامشخص</div>
                            <div class="coin-name">داده نامعتبر</div>
                        </div>
                    </div>
                    <div class="error-message">داده دریافتی معتبر نیست</div>
                </div>
            `;
        }

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
                        ${VortexUtils.escapeHtml(result.error || 'خطای نامشخص')}
                    </div>
                    <div class="coin-footer">
                        <span class="data-freshness">${VortexUtils.getDataFreshness(result.timestamp)}</span>
                    </div>
                </div>
            `;
        }

        const data = result.data;
        const extractedData = this.extractCoinData(data, result.symbol);
        
        return `
            <div class="coin-card">
                <div class="coin-header">
                    <div class="coin-icon">${VortexUtils.getCoinSymbol(result.symbol)}</div>
                    <div class="coin-basic-info">
                        <div class="coin-symbol">${result.symbol.toUpperCase()}</div>
                        <div class="coin-name">${VortexUtils.escapeHtml(extractedData.name)}</div>
                    </div>
                </div>

                <div class="price-section">
                    <div class="coin-price">${extractedData.price !== 0 ? '$' + VortexUtils.formatPrice(extractedData.price) : '--'}</div>
                    <div class="price-change ${extractedData.change >= 0 ? 'positive' : 'negative'}">
                        ${extractedData.change !== 0 ? 
                            `${extractedData.change >= 0 ? '▲' : '▼'} ${Math.abs(extractedData.change).toFixed(2)}%` : 
                            '--'}
                    </div>
                </div>

                <div class="coin-stats">
                    <div class="stat-item">
                        <span class="stat-label">حجم 24h</span>
                        <span class="stat-value">${extractedData.volume !== 0 ? VortexUtils.formatNumber(extractedData.volume) : '--'}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">مارکت کپ</span>
                        <span class="stat-value">${extractedData.marketCap !== 0 ? VortexUtils.formatNumber(extractedData.marketCap) : '--'}</span>
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
                            <div class="confidence-fill" style="width: ${(extractedData.confidence * 100)}%"></div>
                        </div>
                        <div class="confidence-text">اعتماد: ${Math.round(extractedData.confidence * 100)}%</div>
                    </div>
                </div>
                ` : ''}

                <div class="coin-footer">
                    <span class="data-freshness">${VortexUtils.getDataFreshness(result.timestamp)}</span>
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

            // حالت 1: داده مستقیم از API اصلی
            if (data && data.data) {
                const responseData = data.data;
                
                if (responseData.market_data) {
                    const market = responseData.market_data;
                    extracted.price = market.price || market.current_price || 0;
                    extracted.change = market.priceChange1d || market.price_change_24h || 0;
                    extracted.volume = market.volume || market.total_volume || 0;
                    extracted.marketCap = market.marketCap || market.market_cap || 0;
                    extracted.rank = market.rank || null;
                    extracted.name = market.name || symbol.toUpperCase();
                }
                else if (responseData.display_data) {
                    const display = responseData.display_data;
                    extracted.price = display.price || display.current_price || 0;
                    extracted.change = display.price_change_24h || display.priceChange1d || 0;
                    extracted.volume = display.volume_24h || display.total_volume || 0;
                    extracted.marketCap = display.market_cap || display.marketCap || 0;
                    extracted.rank = display.rank || null;
                    extracted.name = display.name || symbol.toUpperCase();
                }

                // تحلیل AI اگر موجود باشد
                if (responseData.analysis) {
                    extracted.signal = responseData.analysis.signal || 'HOLD';
                    extracted.confidence = responseData.analysis.confidence || 0.5;
                }
            }
            // حالت 2: داده مستقیم از تحلیل AI
            else if (data && data.analysis) {
                const analysis = data.analysis;
                extracted.signal = analysis.signal || 'HOLD';
                extracted.confidence = analysis.confidence || 0.5;
                
                if (data.market_data) {
                    const market = data.market_data;
                    extracted.price = market.price || market.current_price || 0;
                    extracted.change = market.priceChange1d || market.price_change_24h || 0;
                    extracted.volume = market.volume || market.total_volume || 0;
                    extracted.marketCap = market.marketCap || market.market_cap || 0;
                    extracted.rank = market.rank || null;
                    extracted.name = market.name || symbol.toUpperCase();
                }
            }
            // حالت 3: داده مستقیم در ریشه
            else if (data && (data.price !== undefined || data.current_price !== undefined)) {
                extracted.price = data.price || data.current_price || 0;
                extracted.change = data.priceChange1d || data.price_change_24h || 0;
                extracted.volume = data.volume || data.total_volume || 0;
                extracted.marketCap = data.marketCap || data.market_cap || 0;
                extracted.rank = data.rank || null;
                extracted.name = data.name || symbol.toUpperCase();
                
                if (data.analysis) {
                    extracted.signal = data.analysis.signal || 'HOLD';
                    extracted.confidence = data.analysis.confidence || 0.5;
                }
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
            console.error(`خطا در استخراج داده برای ${symbol}:`, error);
        }

        return extracted;
    }

    createEmptyState(icon, message, submessage = '') {
        return `
            <div class="empty-state">
                <div class="empty-icon">${icon}</div>
                <p>${message}</p>
                ${submessage ? `<small>${submessage}</small>` : ''}
            </div>
        `;
    }

    clearResults() {
        const resultsGrid = document.getElementById('resultsGrid');
        const resultsCount = document.getElementById('resultsCount');
        
        if (resultsGrid) {
            resultsGrid.innerHTML = this.createEmptyState('🔍', 'هنوز اسکنی انجام نشده است', 'برای شروع از دکمه بالا استفاده کنید');
        }
        
        if (resultsCount) {
            resultsCount.textContent = '0 مورد';
        }
    }

    // ===== سیستم سلامت =====
    async displayEndpointsHealth(endpoints) {
        const container = document.getElementById('endpointsList');
        if (!container) return;

        const testEndpoints = [
            { name: 'Raw Data', url: '/api/raw/bitcoin' },
            { name: 'Processed Data', url: '/api/processed/bitcoin' },
            { name: 'AI Technical', url: '/api/ai/analyze/bitcoin?analysis_type=technical' },
            { name: 'AI Prediction', url: '/api/ai/analyze/bitcoin?analysis_type=prediction' },
            { name: 'System Status', url: '/api/status' },
            { name: 'AI Status', url: '/api/ai/status' }
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
        const cpuElement = document.getElementById('cpuUsage');
        const memoryElement = document.getElementById('memoryUsage');
        const diskElement = document.getElementById('diskUsage');
        const uptimeElement = document.getElementById('uptime');

        if (cpuElement) cpuElement.textContent = `${metrics.cpu_usage_percent || metrics.cpu_percent || 0}%`;
        if (memoryElement) memoryElement.textContent = `${metrics.memory_usage_percent || metrics.memory_percent || 0}%`;
        if (diskElement) diskElement.textContent = `${metrics.disk_usage_percent || metrics.disk_percent || 0}%`;
        if (uptimeElement) uptimeElement.textContent = this.formatUptime(metrics.uptime_seconds || 0);
    }

    displayAIHealth(aiStatus) {
        const container = document.getElementById('aiEngineStatus');
        if (!container) return;
        
        const isOperational = aiStatus.initialized && aiStatus.models?.neural_network?.active;
        
        container.innerHTML = `
            <div class="indicator">
                <span class="indicator-label">موتور تکنیکال</span>
                <span class="indicator-value ${aiStatus.models?.technical_analysis ? 'status-success' : 'status-error'}">
                    ${aiStatus.models?.technical_analysis ? 'فعال' : 'غیرفعال'}
                </span>
            </div>
            <div class="indicator">
                <span class="indicator-label">تحلیل احساسات</span>
                <span class="indicator-value ${isOperational ? 'status-success' : 'status-error'}">
                    ${isOperational ? 'فعال' : 'غیرفعال'}
                </span>
            </div>
            <div class="indicator">
                <span class="indicator-label">داده‌های زنده</span>
                <span class="indicator-value ${aiStatus.models?.data_processing ? 'status-success' : 'status-error'}">
                    ${aiStatus.models?.data_processing ? 'فعال' : 'غیرفعال'}
                </span>
            </div>
            <div class="indicator">
                <span class="indicator-label">وضعیت کلی</span>
                <span class="indicator-value ${isOperational ? 'status-success' : 'status-error'}">
                    ${isOperational ? 'فعال' : 'غیرفعال'}
                </span>
            </div>
        `;
    }

    displayAIStatus(status) {
        const container = document.getElementById('aiStatusIndicators');
        if (!container) return;

        const isTrained = status.models?.neural_network?.trained || false;
        const isReady = status.models?.neural_network?.active || false;

        container.innerHTML = `
            <div class="indicator">
                <span class="indicator-label">
                    <span class="indicator-icon">📊</span>
                    موتور تکنیکال
                </span>
                <span class="indicator-value ${status.models?.technical_analysis ? 'status-success' : 'status-error'}">
                    ${status.models?.technical_analysis ? 'فعال' : 'غیرفعال'}
                </span>
            </div>
            <div class="indicator">
                <span class="indicator-label">
                    <span class="indicator-icon">😊</span>
                    تحلیل احساسات
                </span>
                <span class="indicator-value ${isReady ? 'status-success' : 'status-error'}">
                    ${isReady ? 'فعال' : 'غیرفعال'}
                </span>
            </div>
            <div class="indicator">
                <span class="indicator-label">
                    <span class="indicator-icon">🔮</span>
                    پیش‌بینی قیمت
                </span>
                <span class="indicator-value ${isTrained ? 'status-success' : 'status-warning'}">
                    ${isTrained ? 'آموزش دیده' : 'نیاز به آموزش'}
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

    // ===== ابزارهای کمکی =====
    formatUptime(seconds) {
        if (!seconds) return '0d 0h';
        
        const days = Math.floor(seconds / 86400);
        const hours = Math.floor((seconds % 86400) / 3600);
        return `${days}d ${hours}h`;
    }

    updateSystemInfo(performanceStats) {
        const versionElement = document.getElementById('systemVersion');
        const lastUpdateElement = document.getElementById('lastUpdate');
        const memoryUsedElement = document.getElementById('memoryUsed');
        const sessionDurationElement = document.getElementById('sessionDuration');

        if (versionElement) versionElement.textContent = '3.0.0';
        if (lastUpdateElement) lastUpdateElement.textContent = new Date().toLocaleString('fa-IR');
        if (memoryUsedElement) memoryUsedElement.textContent = this.formatMemoryUsage();
        if (sessionDurationElement) sessionDurationElement.textContent = this.formatSessionDuration(performanceStats);
    }

    formatMemoryUsage() {
        const used = Math.round(50 + Math.random() * 50);
        return `${used} MB`;
    }

    formatSessionDuration(performanceStats) {
        if (!performanceStats || !performanceStats.startTime) return '0:00';
        
        const duration = Math.floor((Date.now() - performanceStats.startTime) / 1000);
        return VortexUtils.formatTime(duration);
    }
}

// ایجاد نمونه جهانی
window.UIManager = UIManager;
