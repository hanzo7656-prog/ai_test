// اپلیکیشن اصلی
let aiMode = false;
let apiStatus = 'checking';
let selectedSymbols = [];

// مدیریت منوی موبایل
function toggleMobileMenu() {
    const menu = document.getElementById('navMenu');
    menu.classList.toggle('active');
}

function closeMobileMenu() {
    const menu = document.getElementById('navMenu');
    menu.classList.remove('active');
}

// تست اتصال API
async function checkAPIStatus() {
    try {
        const response = await fetch('/api/system/status');
        if (response.ok) {
            const data = await response.json();
            apiStatus = data.status === 'operational' ? 'connected' : 'disconnected';
            updateStatusIndicator();
            return true;
        }
    } catch (error) {
        apiStatus = 'disconnected';
        updateStatusIndicator();
    }
    return false;
}

function updateStatusIndicator() {
    const indicator = document.getElementById('statusIndicator');
    const text = document.getElementById('statusText');
    
    if (apiStatus === 'connected') {
        indicator.className = 'status-indicator';
        text.textContent = 'متصل';
    } else {
        indicator.className = 'status-indicator offline';
        text.textContent = 'قطع';
    }
}

// بارگذاری بخش‌ها
async function loadSection(section) {
    // حذف event از پارامتر و استفاده از section مستقیماً
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });
    
    // پیدا کردن لینک فعال
    const activeLink = document.querySelector(`[onclick="loadSection('${section}')"]`);
    if (activeLink) {
        activeLink.classList.add('active');
    }

    document.getElementById('content').innerHTML = `
        <div class="loading">
            <div class="loading-spinner"></div>
            <p>در حال بارگذاری...</p>
        </div>
    `;

    try {
        let content = '';
        switch (section) {
            case 'dashboard': 
                content = await loadDashboard(); 
                break;
            case 'scan': 
                content = await loadScan(); 
                break;
            case 'health': 
                content = await loadHealth(); 
                break;
            case 'settings': 
                content = await loadSettings(); 
                break;
            default: 
                content = await loadDashboard();
        }
        document.getElementById('content').innerHTML = content;
    } catch (error) {
        document.getElementById('content').innerHTML = `
            <div class="error-message">
                <h3>خطا در بارگذاری</h3>
                <p>${error.message}</p>
            </div>
        `;
    }
    
    // بستن منوی موبایل بعد از کلیک
    closeMobileMenu();
}

// توابع load برای بخش‌های مختلف
async function loadDashboard() {
    return `
        <div class="card">
            <div class="card-header">
                <h2 class="card-title">📊 داشبورد VortexAI</h2>
            </div>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">${cacheManager.getStats().memory.count}</div>
                    <div class="metric-label">آیتم در کش</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${apiStatus === 'connected' ? '🟢' : '🔴'}</div>
                    <div class="metric-label">وضعیت API</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${optimizedScanner.top100Symbols.length}</div>
                    <div class="metric-label">ارز پشتیبانی شده</div>
                </div>
            </div>
            <div class="welcome-message">
                <div class="welcome-card">
                    <h1>VortexAI</h1>
                    <p>سیستم تحلیل هوشمند بازار ارز دیجیتال</p>
                    <div class="welcome-stats">
                        <div class="stat">اسکن 100 ارز برتر</div>
                        <div class="stat">تحلیل هوش مصنوعی</div>
                        <div class="stat">پردازش Real-time</div>
                    </div>
                </div>
            </div>
        </div>
    `;
}

async function loadScan() {
    return `
        <div class="card">
            <div class="card-header">
                <h2 class="card-title">🔍 اسکن بازار ارزهای دیجیتال</h2>
                <div class="cache-stats">
                    <small>کش: ${cacheManager.getStats().memory.count} آیتم</small>
                </div>
            </div>

            <!-- کنترل حالت -->
            <div class="mode-toggle">
                <div class="mode-option ${!aiMode ? 'active' : ''}" onclick="setScanMode(false)">
                    📊 Manual (داده بهینه)
                </div>
                <div class="mode-option ${aiMode ? 'active' : ''}" onclick="setScanMode(true)">
                    🤖 AI (داده کامل)
                </div>
            </div>

            <!-- منوی همبرگری برای فیلتر تعداد ارز -->
            <div class="control-group">
                <h3 class="control-title">فیلتر تعداد ارز</h3>
                <div class="hamburger-menu">
                    <button class="btn-outline" onclick="toggleCurrencyFilter()" style="width: 100%;">
                        ☰ انتخاب سریع تعداد ارز
                    </button>
                    <div id="currencyFilterMenu" class="filter-menu">
                        <div class="filter-option" onclick="selectTop10()">🔢 10 ارز برتر</div>
                        <div class="filter-option" onclick="selectTop50()">🔢 50 ارز برتر</div>
                        <div class="filter-option" onclick="selectTop100()">🔢 100 ارز برتر</div>
                        <div class="filter-option" onclick="clearSelection()">🗑️ پاک کردن انتخاب</div>
                    </div>
                </div>
            </div>

            <!-- انتخاب ارزها -->
            <div class="control-group">
                <h3 class="control-title">انتخاب ارزها</h3>
                <div class="multi-select-container">
                    <textarea 
                        class="multi-select" 
                        id="symbolsSelector"
                        placeholder="نام ارزها را وارد کنید (هر خط یک ارز) یا خالی بگذارید برای اسکن 100 ارز برتر"
                        oninput="updateSelectedSymbols(this.value)"
                    >${selectedSymbols.join('\n')}</textarea>
                    <div class="selected-count" id="selectedCount">
                        ${selectedSymbols.length} ارز
                    </div>
                </div>
                <div style="display: flex; gap: 0.5rem; margin-top: 0.5rem;">
                    <button class="btn-outline btn-sm" onclick="selectTop10()">10 ارز برتر</button>
                    <button class="btn-outline btn-sm" onclick="selectTop50()">50 ارز برتر</button>
                    <button class="btn-outline btn-sm" onclick="selectTop100()">100 ارز برتر</button>
                    <button class="btn-outline btn-sm" onclick="clearSelection()">پاک کردن</button>
                </div>
            </div>

            <!-- دکمه اسکن هوشمند -->
            <div class="control-group">
                <button class="btn" onclick="startSmartScan()" style="width: 100%; padding: 1rem;">
                    🚀 شروع اسکن هوشمند
                </button>
                <div style="text-align: center; margin-top: 0.5rem;">
                    <small style="color: var(--text-light);">
                        ${getScanDescription()}
                    </small>
                </div>
            </div>

            <!-- نتایج -->
            <div class="card">
                <div class="card-header">
                    <h3 class="card-title">نتایج اسکن</h3>
                    <div style="display: flex; gap: 0.5rem; align-items: center;">
                        <span id="resultsCount">0 ارز</span>
                        <button class="btn-outline btn-sm" onclick="clearResults()">پاکسازی</button>
                        <button class="btn-outline btn-sm" onclick="exportResults()">خروجی</button>
                    </div>
                </div>
                <div class="symbols-grid" id="scanResults">
                    <div class="no-results">
                        <p>هنوز اسکنی انجام نشده است</p>
                        <small>برای شروع اسکن از دکمه بالا استفاده کنید</small>
                    </div>
                </div>
            </div>
        </div>
    `;
}

async function loadHealth() {
    return `
        <div class="card">
            <div class="card-header">
                <h2 class="card-title">❤️ سلامت سیستم</h2>
            </div>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">${apiStatus === 'connected' ? '🟢' : '🔴'}</div>
                    <div class="metric-label">اتصال API</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${cacheManager.getStats().memory.count}</div>
                    <div class="metric-label">کش فعال</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${navigator.onLine ? '🟢' : '🔴'}</div>
                    <div class="metric-label">اتصال اینترنت</div>
                </div>
            </div>
            <div class="card">
                <h3>لاگ سیستم</h3>
                <div class="logs-container">
                    <div class="log-entry">
                        <span class="log-time">${new Date().toLocaleTimeString('fa-IR')}</span>
                        <span class="log-level level-info">INFO</span>
                        <span class="log-message">سیستم با موفقیت بارگذاری شد</span>
                    </div>
                    <div class="log-entry">
                        <span class="log-time">${new Date().toLocaleTimeString('fa-IR')}</span>
                        <span class="log-level level-success">SUCCESS</span>
                        <span class="log-message">اتصال به API برقرار شد</span>
                    </div>
                </div>
            </div>
        </div>
    `;
}

async function loadSettings() {
    return `
        <div class="card">
            <div class="card-header">
                <h2 class="card-title">⚙️ تنظیمات</h2>
            </div>
            <div class="control-group">
                <h3 class="control-title">تنظیمات اسکن</h3>
                <div style="margin-bottom: 1rem;">
                    <label>سایز دسته‌ها:</label>
                    <select id="batchSize" onchange="updateBatchSize(this.value)">
                        <option value="10">10 ارز</option>
                        <option value="25" selected>25 ارز</option>
                        <option value="50">50 ارز</option>
                    </select>
                </div>
                <div style="margin-bottom: 1rem;">
                    <label>زمان کش (دقیقه):</label>
                    <select id="cacheTTL" onchange="updateCacheTTL(this.value)">
                        <option value="1">1 دقیقه</option>
                        <option value="5" selected>5 دقیقه</option>
                        <option value="10">10 دقیقه</option>
                    </select>
                </div>
            </div>
            <div class="control-group">
                <h3 class="control-title">مدیریت کش</h3>
                <button class="btn-outline" onclick="clearAllCache()">پاکسازی کش</button>
                <button class="btn-outline" onclick="showCacheStats()">نمایش آمار کش</button>
            </div>
        </div>
    `;
}

// توابع کمکی
function setScanMode(isAI) {
    aiMode = isAI;
    // آپدیت UI
    const options = document.querySelectorAll('.mode-option');
    options[0].classList.toggle('active', !isAI);
    options[1].classList.toggle('active', isAI);
}

function updateSelectedSymbols(text) {
    selectedSymbols = text.split('\n')
        .map(s => s.trim())
        .filter(s => s.length > 0);
    
    const countElement = document.getElementById('selectedCount');
    if (countElement) {
        countElement.textContent = selectedSymbols.length + ' ارز';
    }
}

function toggleCurrencyFilter() {
    const menu = document.getElementById('currencyFilterMenu');
    if (menu) {
        menu.style.display = menu.style.display === 'none' ? 'block' : 'none';
    }
}

function selectTop10() {
    selectedSymbols = optimizedScanner.top100Symbols.slice(0, 10);
    updateSymbolsSelector();
    toggleCurrencyFilter();
}

function selectTop50() {
    selectedSymbols = optimizedScanner.top100Symbols.slice(0, 50);
    updateSymbolsSelector();
    toggleCurrencyFilter();
}

function selectTop100() {
    selectedSymbols = optimizedScanner.top100Symbols.slice(0, 100);
    updateSymbolsSelector();
    toggleCurrencyFilter();
}

function clearSelection() {
    selectedSymbols = [];
    updateSymbolsSelector();
    toggleCurrencyFilter();
}

function updateSymbolsSelector() {
    const selector = document.getElementById('symbolsSelector');
    if (selector) {
        selector.value = selectedSymbols.join('\n');
        updateSelectedSymbols(selector.value);
    }
}

function getScanDescription() {
    if (selectedSymbols.length === 0) {
        return 'اسکن 100 ارز برتر بازار';
    } else if (selectedSymbols.length === 1) {
        return `اسکن تکی ${selectedSymbols[0]}`;
    } else {
        return `اسکن دسته‌ای ${selectedSymbols.length} ارز انتخابی`;
    }
}

function startSmartScan() {
    optimizedScanner.smartScan(selectedSymbols, aiMode);
}

function clearResults() {
    const container = document.getElementById('scanResults');
    if (container) {
        container.innerHTML = `
            <div class="no-results">
                <p>نتایج پاکسازی شد</p>
            </div>
        `;
    }
    const countElement = document.getElementById('resultsCount');
    if (countElement) {
        countElement.textContent = '0 ارز';
    }
}

function exportResults() {
    alert('قابلیت خروجی به زودی اضافه می‌شود');
}

function cancelScan() {
    optimizedScanner.cancelScan();
}

function updateBatchSize(size) {
    optimizedScanner.batchSize = parseInt(size);
    alert(`سایز دسته به ${size} ارز تغییر کرد`);
}

function updateCacheTTL(ttl) {
    alert(`زمان کش به ${ttl} دقیقه تغییر کرد`);
}

function clearAllCache() {
    cacheManager.clear();
    alert('کش با موفقیت پاکسازی شد');
}

function showCacheStats() {
    const stats = cacheManager.getStats();
    alert(`آمار کش:\nحافظه: ${stats.memory.count} آیتم\nفایل: ${stats.localStorage.count} آیتم`);
}

// بستن منوی فیلتر وقتی کلیک خارج شود
document.addEventListener('click', function(event) {
    const menu = document.getElementById('currencyFilterMenu');
    const button = document.querySelector('.hamburger-menu button');
    
    if (menu && button && !menu.contains(event.target) && !button.contains(event.target)) {
        menu.style.display = 'none';
    }
});

// راه‌اندازی اولیه
window.addEventListener('load', async function() {
    await checkAPIStatus();
    loadSection('dashboard');
    setInterval(checkAPIStatus, 30000);
});

// مدیریت خطاها
window.addEventListener('error', function(event) {
    console.error('خطا:', event.error);
});

window.addEventListener('unhandledrejection', function(event) {
    console.error('Promise رد شده:', event.reason);
});
