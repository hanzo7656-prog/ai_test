// مدیریت صفحه بازار
let currentPage = 1;
let currentSort = 'rank';
let currentSortDir = 'asc';
let currentLimit = 20;
let allCoins = [];

document.addEventListener('DOMContentLoaded', function() {
    loadMarketData();
    setupEventListeners();
    loadWebSocketData();
});

// بارگذاری داده‌های بازار
async function loadMarketData() {
    try {
        showLoadingSkeleton();
        
        const data = await apiCall(`/coins/list?limit=${currentLimit}&page=${currentPage}&sort_by=${currentSort}&sort_dir=${currentSortDir}`);
        
        if (data && data.result) {
            allCoins = data.result;
            renderCoinsTable(allCoins);
            updatePagination();
        }
    } catch (error) {
        handleError(error, 'بارگذاری داده‌های بازار');
    }
}

// رندر جدول کوین‌ها
function renderCoinsTable(coins) {
    const tbody = document.getElementById('coins-table-body');
    
    if (!coins || coins.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="8" class="px-6 py-4 text-center text-gray-500">
                    داده‌ای یافت نشد
                </td>
            </tr>
        `;
        return;
    }
    
    tbody.innerHTML = coins.map((coin, index) => `
        <tr class="hover:bg-gray-50">
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                ${coin.rank || index + 1}
            </td>
            <td class="px-6 py-4 whitespace-nowrap">
                <div class="flex items-center">
                    <div class="flex-shrink-0 h-10 w-10">
                        <img class="h-10 w-10 rounded-full" src="${coin.icon || '/static/images/coin-placeholder.png'}" alt="${coin.name}">
                    </div>
                    <div class="mr-4">
                        <div class="text-sm font-medium text-gray-900">${coin.name}</div>
                        <div class="text-sm text-gray-500">${coin.symbol}</div>
                    </div>
                </div>
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                $${formatNumber(coin.price || 0)}
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm">
                <span class="${getChangeColor(coin.priceChange1d)}">
                    ${formatPercent(coin.priceChange1d)}
                </span>
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                $${formatMarketCap(coin.marketCap)}
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                $${formatMarketCap(coin.volume)}
            </td>
            <td class="px-6 py-4 whitespace-nowrap">
                <button onclick="quickCoinAnalysis('${coin.symbol}')" 
                        class="bg-blue-100 text-blue-600 px-3 py-1 rounded text-xs font-semibold hover:bg-blue-200">
                    تحلیل AI
                </button>
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm font-medium">
                <button onclick="viewCoinDetail('${coin.id || coin.symbol}')" 
                        class="text-blue-600 hover:text-blue-900 mr-4">
                    جزئیات
                </button>
                <button onclick="setAlertForCoin('${coin.symbol}')" 
                        class="text-green-600 hover:text-green-900">
                    هشدار
                </button>
            </td>
        </tr>
    `).join('');
}

// تحلیل سریع کوین
async function quickCoinAnalysis(symbol) {
    try {
        const analysis = await apiCall(`/ai/analysis?symbols=${symbol}&period=1d`);
        showNotification(`تحلیل ${symbol}: ${analysis.analysis.signal}`, 'success');
    } catch (error) {
        handleError(error, `تحلیل ${symbol}`);
    }
}

// مشاهده جزئیات کوین
function viewCoinDetail(coinId) {
    window.location.href = `/coin/${coinId}`;
}

// تنظیم هشدار برای کوین
function setAlertForCoin(symbol) {
    document.getElementById('alert-symbol').value = symbol;
    document.getElementById('alerts-page-link').click();
}

// بارگذاری داده‌های WebSocket
async function loadWebSocketData() {
    try {
        const wsData = await apiCall('/websocket/pairs/active');
        renderWebSocketData(wsData.active_pairs || []);
    } catch (error) {
        console.error('Error loading WebSocket data:', error);
    }
}

// رندر داده‌های WebSocket
function renderWebSocketData(pairs) {
    const container = document.getElementById('websocket-data');
    
    if (!pairs || pairs.length === 0) {
        container.innerHTML = '<div class="text-gray-500">هیچ داده لحظه‌ای در دسترس نیست</div>';
        return;
    }
    
    // فقط 12 جفت اول را نمایش بده
    const displayPairs = pairs.slice(0, 12);
    
    container.innerHTML = displayPairs.map(pair => `
        <div class="bg-gray-50 p-3 rounded-lg text-center">
            <div class="font-semibold text-sm">${pair.toUpperCase()}</div>
            <div class="text-green-600 font-bold text-lg" id="ws-price-${pair}">
                ...
            </div>
            <div class="text-xs text-gray-500">زنده</div>
        </div>
    `).join('');
}

// تنظیم event listeners
function setupEventListeners() {
    // جستجو
    document.getElementById('search-coins').addEventListener('input', function(e) {
        const searchTerm = e.target.value.toLowerCase();
        const filteredCoins = allCoins.filter(coin => 
            coin.name.toLowerCase().includes(searchTerm) || 
            coin.symbol.toLowerCase().includes(searchTerm)
        );
        renderCoinsTable(filteredCoins);
    });
    
    // مرتب‌سازی
    document.getElementById('sort-by').addEventListener('change', function(e) {
        currentSort = e.target.value;
        loadMarketData();
    });
    
    document.getElementById('sort-dir').addEventListener('change', function(e) {
        currentSortDir = e.target.value;
        loadMarketData();
    });
    
    // محدودیت نمایش
    document.getElementById('limit').addEventListener('change', function(e) {
        currentLimit = parseInt(e.target.value);
        currentPage = 1;
        loadMarketData();
    });
    
    // صفحه‌بندی
    document.getElementById('prev-page').addEventListener('click', function() {
        if (currentPage > 1) {
            currentPage--;
            loadMarketData();
        }
    });
    
    document.getElementById('next-page').addEventListener('click', function() {
        currentPage++;
        loadMarketData();
    });
}

// نمایش اسکلت بارگذاری
function showLoadingSkeleton() {
    const tbody = document.getElementById('coins-table-body');
    tbody.innerHTML = `
        ${Array.from({length: 10}, () => `
        <tr>
            <td class="px-6 py-4 whitespace-nowrap"><div class="h-4 bg-gray-200 rounded w-8 animate-pulse"></div></td>
            <td class="px-6 py-4 whitespace-nowrap">
                <div class="flex items-center">
                    <div class="h-10 w-10 bg-gray-200 rounded-full animate-pulse"></div>
                    <div class="mr-4">
                        <div class="h-4 bg-gray-200 rounded w-20 mb-2 animate-pulse"></div>
                        <div class="h-3 bg-gray-200 rounded w-12 animate-pulse"></div>
                    </div>
                </div>
            </td>
            <td class="px-6 py-4"><div class="h-4 bg-gray-200 rounded w-16 animate-pulse"></div></td>
            <td class="px-6 py-4"><div class="h-4 bg-gray-200 rounded w-12 animate-pulse"></div></td>
            <td class="px-6 py-4"><div class="h-4 bg-gray-200 rounded w-20 animate-pulse"></div></td>
            <td class="px-6 py-4"><div class="h-4 bg-gray-200 rounded w-20 animate-pulse"></div></td>
            <td class="px-6 py-4"><div class="h-6 bg-gray-200 rounded w-16 animate-pulse"></div></td>
            <td class="px-6 py-4"><div class="h-6 bg-gray-200 rounded w-24 animate-pulse"></div></td>
        </tr>
        `).join('')}
    `;
}

// به‌روزرسانی صفحه‌بندی
function updatePagination() {
    document.getElementById('current-page').textContent = currentPage;
    document.getElementById('total-pages').textContent = Math.ceil(100 / currentLimit); // فرضی
    
    document.getElementById('prev-page').disabled = currentPage === 1;
    document.getElementById('next-page').disabled = currentPage * currentLimit >= 100; // فرضی
}

// utility functions
function formatNumber(num) {
    if (!num) return '0';
    return new Intl.NumberFormat('en-US', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 6
    }).format(num);
}

function formatPercent(percent) {
    if (!percent) return '0%';
    return `${percent > 0 ? '+' : ''}${percent.toFixed(2)}%`;
}

function formatMarketCap(cap) {
    if (!cap) return '0';
    if (cap >= 1e9) return (cap / 1e9).toFixed(2) + 'B';
    if (cap >= 1e6) return (cap / 1e6).toFixed(2) + 'M';
    return formatNumber(cap);
}

function getChangeColor(change) {
    if (!change) return 'text-gray-500';
    return change > 0 ? 'text-green-600' : 'text-red-600';
}
