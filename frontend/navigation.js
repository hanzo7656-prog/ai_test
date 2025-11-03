// navigation.js
class NavigationManager {
    constructor() {
        this.currentSection = 'dashboard';
        this.init();
    }

    init() {
        this.bindEvents();
        this.loadInitialSection();
    }

    bindEvents() {
        // رویدادهای کلیک برای منوها
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const target = e.currentTarget.dataset.section;
                this.navigateTo(target);
            });
        });

        // رویداد برای زیرمنوها
        document.querySelectorAll('.submenu-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const target = e.currentTarget.dataset.route;
                this.loadSection(target);
            });
        });

        // سوئیچ بین حالت‌های اسکن
        document.querySelectorAll('.scan-mode-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                const mode = e.currentTarget.dataset.mode;
                this.switchScanMode(mode);
            });
        });
    }

    async navigateTo(section) {
        try {
            this.showLoading();
            
            // غیرفعال کردن منوی قبلی
            document.querySelectorAll('.nav-item').forEach(item => {
                item.classList.remove('active');
            });
            
            // فعال کردن منوی جدید
            document.querySelector(`[data-section="${section}"]`).classList.add('active');
            
            await this.loadSection(section);
            this.hideLoading();
            
        } catch (error) {
            this.handleNavigationError(error);
        }
    }

    async loadSection(section) {
        try {
            const response = await fetch(`/api/${this.getApiEndpoint(section)}`);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.renderSection(section, data);
            this.currentSection = section;
            
        } catch (error) {
            this.handleSectionError(section, error);
        }
    }

    getApiEndpoint(section) {
        const endpoints = {
            'ai-scan': 'ai/scan',
            'ai-analysis': 'ai/analysis',
            'ai-technical': 'ai/technical/analysis',
            'ai-quick': 'ai/analysis/quick',
            'system-dashboard': 'system/dashboard',
            'system-health': 'system/health',
            'system-alerts': 'system/alerts',
            'system-metrics': 'system/metrics',
            'system-tests': 'system/tests/run',
            'system-logs': 'system/logs',
            'settings-cache': 'system/cache/clear',
            'settings-ai': 'ai/train',
            'settings-debug': 'system/debug'
        };

        return endpoints[section] || 'system/health';
    }

    renderSection(section, data) {
        const contentArea = document.getElementById('main-content');
        
        if (!contentArea) {
            throw new Error('Content area not found');
        }

        contentArea.innerHTML = this.generateContent(section, data);
        this.initializeSectionScripts(section);
    }

    generateContent(section, data) {
        const templates = {
            'ai-scan': this.generateAIScanContent(data),
            'system-dashboard': this.generateDashboardContent(data),
            'system-health': this.generateHealthContent(data),
            // ... سایر تمپلیت‌ها
        };

        return templates[section] || this.generateErrorContent('Section template not found');
    }

    generateAIScanContent(data) {
        return `
            <div class="section-content">
                <h2>اسکن بازار</h2>
                <div class="scan-mode-selector">
                    <button class="scan-mode-btn active" data-mode="ai">حالت هوشمند</button>
                    <button class="scan-mode-btn" data-mode="manual">حالت دستی</button>
                </div>
                <div id="scan-results">
                    ${this.renderScanResults(data)}
                </div>
            </div>
        `;
    }

    generateDashboardContent(data) {
        if (!data || data.error) {
            return this.generateErrorContent(data?.error || 'No data available');
        }

        return `
            <div class="section-content">
                <h2>داشبورد سیستم</h2>
                <div class="health-cards">
                    <div class="health-card ${data.system_health?.status || 'unknown'}">
                        <h3>وضعیت سیستم</h3>
                        <div class="health-score">${data.health_score || 0}%</div>
                    </div>
                    <div class="health-card ${data.api_health?.status || 'unknown'}">
                        <h3>وضعیت API</h3>
                        <div class="health-score">${data.api_health?.healthy_endpoints || 0}/${data.api_health?.total_endpoints || 0}</div>
                    </div>
                </div>
            </div>
        `;
    }

    renderScanResults(data) {
        if (!data || data.error) {
            return `<div class="error-message">خطا در دریافت داده: ${data?.error || 'Unknown error'}</div>`;
        }

        if (!data.scan_results || data.scan_results.length === 0) {
            return '<div class="no-data">داده‌ای برای نمایش وجود ندارد</div>';
        }

        return data.scan_results.map(result => `
            <div class="scan-result-item">
                <div class="symbol">${result.symbol}</div>
                <div class="price">${result.current_price}</div>
                <div class="signal ${result.ai_signal?.primary_signal?.toLowerCase() || 'hold'}">
                    ${result.ai_signal?.primary_signal || 'HOLD'}
                </div>
            </div>
        `).join('');
    }

    switchScanMode(mode) {
        document.querySelectorAll('.scan-mode-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-mode="${mode}"]`).classList.add('active');
        
        // بارگذاری داده‌ها بر اساس حالت انتخاب شده
        this.loadScanData(mode);
    }

    async loadScanData(mode) {
        try {
            const requestBody = {
                symbols: ["BTC", "ETH", "ADA", "SOL"],
                scan_mode: mode,
                timeframe: "1h"
            };

            const response = await fetch('/api/ai/scan', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody)
            });

            if (!response.ok) {
                throw new Error(`Scan failed: ${response.status}`);
            }

            const data = await response.json();
            this.updateScanResults(data);
            
        } catch (error) {
            this.handleScanError(error);
        }
    }

    updateScanResults(data) {
        const resultsContainer = document.getElementById('scan-results');
        if (resultsContainer) {
            resultsContainer.innerHTML = this.renderScanResults(data);
        }
    }

    handleNavigationError(error) {
        console.error('Navigation error:', error);
        this.showError('خطا در تغییر بخش: ' + error.message);
    }

    handleSectionError(section, error) {
        console.error(`Error loading section ${section}:`, error);
        
        const contentArea = document.getElementById('main-content');
        if (contentArea) {
            contentArea.innerHTML = this.generateErrorContent(`خطا در بارگذاری ${section}: ${error.message}`);
        }
    }

    handleScanError(error) {
        console.error('Scan error:', error);
        this.showError('خطا در اسکن بازار: ' + error.message);
    }

    generateErrorContent(message) {
        return `
            <div class="error-container">
                <div class="error-icon">⚠️</div>
                <h3>خطا</h3>
                <p>${message}</p>
                <button onclick="navigationManager.retryLastAction()" class="retry-btn">تلاش مجدد</button>
            </div>
        `;
    }

    showError(message) {
        // نمایش نوتیفیکیشن خطا
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-notification';
        errorDiv.textContent = message;
        document.body.appendChild(errorDiv);

        setTimeout(() => {
            errorDiv.remove();
        }, 5000);
    }

    showLoading() {
        // نمایش ایندیکیتور لودینگ
        document.body.classList.add('loading');
    }

    hideLoading() {
        // پنهان کردن ایندیکیتور لودینگ
        document.body.classList.remove('loading');
    }

    retryLastAction() {
        this.loadSection(this.currentSection);
    }

    initializeSectionScripts(section) {
        // مقداردهی اولیه اسکریپت‌های خاص هر بخش
        if (section.includes('scan')) {
            this.initializeScanScripts();
        }
    }

    initializeScanScripts() {
        // اسکریپت‌های مخصوص بخش اسکن
        document.querySelectorAll('.scan-mode-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                const mode = e.currentTarget.dataset.mode;
                this.switchScanMode(mode);
            });
        });
    }

    loadInitialSection() {
        this.navigateTo('system-dashboard');
    }
}

// ایجاد نمونه گلوبال
const navigationManager = new NavigationManager();
