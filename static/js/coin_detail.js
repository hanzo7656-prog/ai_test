// مدیریت صفحه جزئیات کوین
let currentCoinId = null;

document.addEventListener('DOMContentLoaded', function() {
    loadCoinDetail();
});

// بارگذاری جزئیات کوین
async function loadCoinDetail() {
    const pathParts = window.location.pathname.split('/');
    const coinId = pathParts[pathParts.length - 1];
    
    if (!coinId || coinId === 'coin_detail.html') {
        window.location.href = '/market';
        return;
    }
    
    currentCoinId = coinId;
    
    try {
        const [coinData, chartData, analysisData] = await Promise.all([
            apiCall(`/coins/${coinId}`),
            apiCall(`/coins/${coinId}/charts?period=7d`),
            apiCall(`/ai/analysis?symbols=${coinId}&period=7d`)
        ]);
        
        renderCoinDetail(coinData, chartData, analysisData);
    } catch (error) {
        handleError(error, 'بارگذاری جزئیات کوین');
        showErrorState();
    }
}

// رندر جزئیات کوین
function renderCoinDetail(coinData, chartData, analysisData) {
    const container = document.getElementById('coin-detail-container');
    const skeleton = document.getElementById('coin-detail-skeleton');
    
    skeleton.classList.add('hidden');
    
    if (!coinData || !coinData.result) {
        showErrorState();
        return;
    }
    
    const coin = coinData.result;
    const analysis = analysisData?.analysis_report || {};
    
    container.innerHTML = `
        <!-- هدر کوین -->
        <div class="bg-white rounded-lg shadow-md p-6 mb-6">
            <div class="flex items-center justify-between mb-6">
                <div class="flex items-center">
                    <img src="${coin.icon || '/static/images/coin-placeholder.png'}" 
                         alt="${coin.name}" class="h-16 w-16 rounded-full">
                    <div class="mr-4">
                        <h1 class="text-2xl font-bold text-gray-900">${coin.name}</h1>
                        <p class="text-gray-600">${coin.symbol}</p>
                    </div>
                </div>
                <div class="text-right">
                    <div class="text-3xl font-bold text-gray-900">
                        $${formatNumber(coin.price)}
                    </div>
                    <div class="text-sm ${coin.priceChange1d >= 0 ? 'text-green-600' : 'text-red-600'}">
                        ${coin.priceChange1d >= 0 ? '▲' : '▼'} ${formatPercent(coin.priceChange1d)} (24h)
                    </div>
                </div>
            </div>
            
            <!-- آمار کلی -->
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                <div class="text-center p-4 bg-gray-50 rounded-lg">
                    <div class="text-sm text-gray-600">ارزش بازار</div>
                    <div class="font-semibold text-lg">$${formatMarketCap(coin.marketCap)}</div>
                </div>
                <div class="text-center p-4 bg-gray-50 rounded-lg">
                    <div class="text-sm text-gray-600">حجم 24h</div>
                    <div class="font-semibold text-lg">$${formatMarketCap(coin.volume)}</div>
                </div>
                <div class="text-center p-4 bg-gray-50 rounded-lg">
                    <div class="text-sm text-gray-600">رتبه</div>
                    <div class="font-semibold text-lg">${coin.rank || '—'}</div>
                </div>
                <div class="text-center p-4 bg-gray-50 rounded-lg">
                    <div class="text-sm text-gray-600">تغییر 7d</div>
                    <div class="font-semibold text-lg ${coin.priceChange7d >= 0 ? 'text-green-600' : 'text-red-600'}">
                        ${formatPercent(coin.priceChange7d)}
                    </div>
                </div>
            </div>
            
            <!-- دکمه‌های عمل -->
            <div class="flex space-x-4 space-x-reverse">
                <button onclick="runCoinAnalysis('${coin.symbol}')" 
                        class="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 font-semibold">
                    تحلیل AI پیشرفته
                </button>
                <button onclick="setAlertForCoin('${coin.symbol}')" 
                        class="bg-green-600 text-white px-6 py-2 rounded-lg hover:bg-green-700 font-semibold">
                    تنظیم هشدار
                </button>
                <button onclick="addToWatchlist('${coin.symbol}')" 
                        class="bg-gray-600 text-white px-6 py-2 rounded-lg hover:bg-gray-700 font-semibold">
                    افزودن به واچ‌لیست
                </button>
            </div>
        </div>
        
        <!-- نمودار و تحلیل -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            <!-- نمودار قیمت -->
            <div class="bg-white rounded-lg shadow-md p-6">
                <h3 class="text-xl font-semibold mb-4">نمودار قیمت 7 روزه</h3>
                <canvas id="coinDetailChart" height="300"></canvas>
            </div>
            
            <!-- تحلیل AI -->
            <div class="bg-white rounded-lg shadow-md p-6">
                <h3 class="text-xl font-semibold mb-4">تحلیل هوش مصنوعی</h3>
                ${renderAIAnalysis(analysis)}
            </div>
        </div>
        
        <!-- اطلاعات تکمیلی -->
        <div class="bg-white rounded-lg shadow-md p-6">
            <h3 class="text-xl font-semibold mb-4">اطلاعات تکمیلی</h3>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                    <h4 class="font-semibold mb-3">آمار تکنیکال</h4>
                    <div class="space-y-2">
                        <div class="flex justify-between">
                            <span class="text-gray-600">بالاترین 24h:</span>
                            <span class="font-semibold">$${formatNumber(coin.highPrice)}</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600">پایین‌ترین 24h:</span>
                            <span class="font-semibold">$${formatNumber(coin.lowPrice)}</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600">تغییر 1h:</span>
                            <span class="${coin.priceChange1h >= 0 ? 'text-green-600' : 'text-red-600'} font-semibold">
                                ${formatPercent(coin.priceChange1h)}
                            </span>
                        </div>
                    </div>
                </div>
                <div>
                    <h4 class="font-semibold mb-3">اطلاعات شبکه</h4>
                    <div class="space-y-2">
                        <div class="flex justify-between">
                            <span class="text-gray-600">تعداد معاملات:</span>
                            <span class="font-semibold">${coin.numberOfMarkets || '—'}</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600">صرافی‌ها:</span>
                            <span class="font-semibold">${coin.numberOfExchanges || '—'}</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600">عرضه در گردش:</span>
                            <span class="font-semibold">${formatNumber(coin.availableSupply)}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // ایجاد نمودار
    createCoinDetailChart(chartData);
}

// رندر تحلیل AI
function renderAIAnalysis(analysis) {
    if (!analysis || !analysis.trading_signals) {
        return `
            <div class="text-center py-8 text-gray-500">
                <div class="text-4xl mb-4">🤖</div>
                <p>تحلیل در دسترس نیست</p>
                <button onclick="runCoinAnalysis('${currentCoinId}')" 
                        class="mt-4 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700">
                    اجرای تحلیل
                </button>
            </div>
        `;
    }
    
    const signal = analysis.trading_signals[currentCoinId] || analysis.trading_signals[Object.keys(analysis.trading_signals)[0]];
    
    if (!signal) {
        return '<p class="text-gray-500">تحلیل برای این کوین موجود نیست</p>';
    }
    
    return `
        <div class="space-y-4">
            <div class="flex items-center justify-between">
                <span class="text-lg font-semibold">سیگنال:</span>
                <span class="px-3 py-1 rounded-full text-white font-semibold ${
                    signal.action === 'BUY' ? 'bg-green-500' :
                    signal.action === 'SELL' ? 'bg-red-500' : 'bg-yellow-500'
                }">
                    ${signal.action === 'BUY' ? 'خرید' : signal.action === 'SELL' ? 'فروش' : 'نگهداری'}
                </span>
            </div>
            
            <div class="flex items-center justify-between">
                <span class="text-gray-600">اطمینان سیگنال:</span>
                <span class="font-semibold">${Math.round(signal.confidence * 100)}%</span>
            </div>
            
            <div class="flex items-center justify-between">
                <span class="text-gray-600">سطح ریسک:</span>
                <span class="font-semibold ${
                    signal.risk_level === 'low' ? 'text-green-600' :
                    signal.risk_level === 'medium' ? 'text-yellow-600' : 'text-red-600'
                }">
                    ${signal.risk_level === 'low' ? 'کم' : signal.risk_level === 'medium' ? 'متوسط' : 'زیاد'}
                </span>
            </div>
            
            <div>
                <span class="text-gray-600 block mb-2">دلیل‌پذیری:</span>
                <p class="text-sm text-gray-700">${signal.reasoning || 'تحلیل بر اساس داده‌های تکنیکال و الگوهای بازار'}</p>
            </div>
            
            <div class="pt-4 border-t border-gray-200">
                <button onclick="runAdvancedAnalysis('${currentCoinId}')" 
                        class="w-full bg-purple-600 text-white py-2 rounded hover:bg-purple-700 font-semibold">
                    تحلیل پیشرفته‌تر
                </button>
            </div>
        </div>
    `;
}

// ایجاد نمودار جزئیات کوین
function createCoinDetailChart(chartData) {
    const ctx = document.getElementById('coinDetailChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.result ? chartData.result.map((_, i) => {
                if (i % 10 === 0) return `روز ${i + 1}`;
                return '';
            }) : [],
            datasets: [{
                label: 'قیمت',
                data: chartData.result ? chartData.result.map(item => item.price) : [],
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    rtl: true,
                    callbacks: {
                        label: function(context) {
                            return `قیمت: $${context.parsed.y.toLocaleString()}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    display: true,
                    grid: {
                        display: false
                    }
                },
                y: {
                    display: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    },
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toLocaleString();
                        }
                    }
                }
            }
        }
    });
}

// اجرای تحلیل کوین
async function runCoinAnalysis(symbol) {
    try {
        showNotification('درحال تحلیل...', 'info');
        
        const analysis = await apiCall(`/ai/analysis?symbols=${symbol}&period=7d&analysis_type=comprehensive`);
        
        if (analysis.analysis_report) {
            renderAIAnalysis(analysis.analysis_report);
            showNotification('تحلیل با موفقیت انجام شد', 'success');
        }
    } catch (error) {
        handleError(error, 'اجرای تحلیل');
    }
}

// اجرای تحلیل پیشرفته
async function runAdvancedAnalysis(symbol) {
    try {
        const analysis = await apiCall(`/ai/analysis?symbols=${symbol}&period=30d&analysis_type=comprehensive&train_model=true`);
        
        if (analysis.analysis_report) {
            // نمایش نتایج در یک مودال یا صفحه جدید
            showAdvancedAnalysisModal(analysis.analysis_report);
        }
    } catch (error) {
        handleError(error, 'اجرای تحلیل پیشرفته');
    }
}

// افزودن به واچ‌لیست
function addToWatchlist(symbol) {
    let watchlist = JSON.parse(localStorage.getItem('watchlist') || '[]');
    
    if (!watchlist.includes(symbol)) {
        watchlist.push(symbol);
        localStorage.setItem('watchlist', JSON.stringify(watchlist));
        showNotification(`${symbol} به واچ‌لیست اضافه شد`, 'success');
    } else {
        showNotification(`${symbol} قبلاً در واچ‌لیست موجود است`, 'info');
    }
}

// نمایش حالت خطا
function showErrorState() {
    const container = document.getElementById('coin-detail-container');
    const skeleton = document.getElementById('coin-detail-skeleton');
    
    skeleton.classList.add('hidden');
    
    container.innerHTML = `
        <div class="bg-white rounded-lg shadow-md p-8 text-center">
            <div class="text-6xl mb-4">😕</div>
            <h2 class="text-2xl font-bold text-gray-800 mb-4">کوین یافت نشد</h2>
            <p class="text-gray-600 mb-6">متأسفانه اطلاعات این ارز دیجیتال در دسترس نیست</p>
            <button onclick="window.location.href='/market'" 
                    class="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 font-semibold">
                بازگشت به بازار
            </button>
        </div>
    `;
}
