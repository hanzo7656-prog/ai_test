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
            apiStatus = 'connected';
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
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });
    event.target.classList.add('active');

    document.getElementById('content').innerHTML = `
        <div class="loading">
            <div class="loading-spinner"></div>
            <p>در حال بارگذاری...</p>
        </div>
    `;

    try {
        let content = '';
        switch (section) {
            case 'dashboard': content = await loadDashboard(); break;
            case 'scan': content = await loadScan(); break;
            case 'health': content = await loadHealth(); break;
            case 'settings': content = await loadSettings(); break;
            default: content = await loadDashboard();
        }
        document.getElementById('content').innerHTML = content;
    } catch (error) {
        showError('خطا در بارگذاری', error.message);
    }
}

// صفحه اسکن بهینه‌شده
async function loadScan() {
    return `
        <div class="card">
            <div class="card-header">
                <h2 class="card-title">اسکن بازار ارزهای دیجیتال</h2>
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
                    <button class="btn-outline btn-sm" onclick="clearSelection()">پاک کردن</button>
                </div>
            </div>

            <!-- دکمه اسکن هوشمند -->
            <div class="control-group">
                <button class="btn btn-success" onclick="startSmartScan()" style="width: 100%; padding: 1rem;">
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
    
    document.getElementById('selectedCount').textContent = selectedSymbols.length + ' ارز';
}

function selectTop10() {
    selectedSymbols = optimizedScanner.top100Symbols.slice(0, 10);
    updateSymbolsSelector();
}

function selectTop50() {
    selectedSymbols = optimizedScanner.top100Symbols.slice(0, 50);
    updateSymbolsSelector();
}

function clearSelection() {
    selectedSymbols = [];
    updateSymbolsSelector();
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
    document.getElementById('scanResults').innerHTML = `
        <div class="no-results">
            <p>نتایج پاکسازی شد</p>
        </div>
    `;
    document.getElementById('resultsCount').textContent = '0 ارز';
}

function exportResults() {
    alert('قابلیت خروجی به زودی اضافه می‌شود');
}

function cancelScan() {
    optimizedScanner.cancelScan();
}

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
