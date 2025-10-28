// مدیریت هشدارها
let alerts = [];
let alertHistory = [];

document.addEventListener('DOMContentLoaded', function() {
    loadAlerts();
    loadAlertHistory();
    setupAlertForm();
});

// بارگذاری هشدارهای فعال
async function loadAlerts() {
    try {
        const data = await apiCall('/alerts/list');
        
        if (data && data.alerts) {
            alerts = data.alerts;
            renderAlertsList();
        }
    } catch (error) {
        handleError(error, 'بارگذاری هشدارها');
    }
}

// رندر لیست هشدارها
function renderAlertsList() {
    const container = document.getElementById('alerts-list');
    const noAlerts = document.getElementById('no-alerts');
    
    if (!alerts || alerts.length === 0) {
        container.innerHTML = '';
        noAlerts.classList.remove('hidden');
        return;
    }
    
    noAlerts.classList.add('hidden');
    
    container.innerHTML = alerts.map(alert => `
        <div class="border border-gray-200 rounded-lg p-4 mb-4 hover:bg-gray-50 transition-colors">
            <div class="flex justify-between items-start mb-3">
                <div>
                    <h4 class="font-semibold text-lg">${alert.symbol}</h4>
                    <p class="text-sm text-gray-600">${formatAlertCondition(alert)}</p>
                </div>
                <div class="flex space-x-2 space-x-reverse">
                    <span class="bg-green-100 text-green-800 px-2 py-1 rounded text-xs font-semibold">
                        فعال
                    </span>
                    <button onclick="deleteAlert('${alert.id}')" 
                            class="text-red-600 hover:text-red-800 text-sm">
                        حذف
                    </button>
                </div>
            </div>
            
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                    <span class="text-gray-500">قیمت فعلی:</span>
                    <div class="font-semibold" id="current-price-${alert.symbol}">
                        درحال بارگذاری...
                    </div>
                </div>
                <div>
                    <span class="text-gray-500">هدف:</span>
                    <div class="font-semibold">$${alert.target_price}</div>
                </div>
                <div>
                    <span class="text-gray-500">نوع:</span>
                    <div class="font-semibold">${formatAlertType(alert.alert_type)}</div>
                </div>
                <div>
                    <span class="text-gray-500">ایجاد شده:</span>
                    <div class="font-semibold">${formatDate(alert.created_at)}</div>
                </div>
            </div>
            
            <div class="mt-3 pt-3 border-t border-gray-200">
                <div class="flex items-center justify-between">
                    <span class="text-sm text-gray-600">وضعیت:</span>
                    <span id="alert-status-${alert.symbol}" class="text-sm font-semibold text-yellow-600">
                        در انتظار فعال‌سازی
                    </span>
                </div>
            </div>
        </div>
    `).join('');
    
    // بارگذاری قیمت‌های جاری برای هشدارها
    alerts.forEach(alert => {
        loadCurrentPriceForAlert(alert.symbol);
    });
}

// بارگذاری قیمت جاری برای هشدار
async function loadCurrentPriceForAlert(symbol) {
    try {
        const coinData = await apiCall(`/coins/${symbol}`);
        
        if (coinData && coinData.result) {
            const currentPrice = coinData.result.price;
            const element = document.getElementById(`current-price-${symbol}`);
            const statusElement = document.getElementById(`alert-status-${symbol}`);
            
            if (element) {
                element.textContent = `$${formatNumber(currentPrice)}`;
            }
            
            if (statusElement) {
                // بررسی وضعیت هشدار
                const alertObj = alerts.find(a => a.symbol === symbol);
                if (alertObj && checkAlertCondition(alertObj, currentPrice)) {
                    statusElement.textContent = 'فعال شده!';
                    statusElement.className = 'text-sm font-semibold text-green-600';
                    triggerAlertNotification(alertObj, currentPrice);
                }
            }
        }
    } catch (error) {
        console.error(`Error loading price for ${symbol}:`, error);
    }
}

// بررسی شرط هشدار
function checkAlertCondition(alert, currentPrice) {
    switch (alert.condition) {
        case 'above':
            return currentPrice >= alert.target_price;
        case 'below':
            return currentPrice <= alert.target_price;
        case 'change_up':
            // پیاده‌سازی تغییر درصدی
            return false;
        case 'change_down':
            // پیاده‌سازی تغییر درصدی
            return false;
        default:
            return false;
    }
}

// فعال‌سازی نوتیفیکیشن هشدار
function triggerAlertNotification(alert, currentPrice) {
    const message = `هشدار ${alert.symbol}: قیمت به $${formatNumber(currentPrice)} رسید!`;
    showNotification(message, 'success');
    
    // می‌توانید صدا یا ویبره هم اضافه کنید
    if ('vibrate' in navigator) {
        navigator.vibrate([200, 100, 200]);
    }
}

// فرمت شرط هشدار
function formatAlertCondition(alert) {
    const conditions = {
        'above': `بالاتر از $${formatNumber(alert.target_price)}`,
        'below': `پایین‌تر از $${formatNumber(alert.target_price)}`,
        'change_up': `افزایش بیش از ${alert.target_price}%`,
        'change_down': `کاهش بیش از ${alert.target_price}%`
    };
    
    return conditions[alert.condition] || 'شرط نامشخص';
}

// فرمت نوع هشدار
function formatAlertType(type) {
    const types = {
        'price': 'قیمتی',
        'volume': 'حجمی',
        'rsi': 'RSI'
    };
    
    return types[type] || type;
}

// تنظیم فرم هشدار
function setupAlertForm() {
    const form = document.getElementById('alert-form');
    
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = {
            symbol: document.getElementById('alert-symbol').value.toUpperCase(),
            condition: document.getElementById('alert-condition').value,
            target_price: parseFloat(document.getElementById('alert-target').value),
            alert_type: document.getElementById('alert-type').value
        };
        
        try {
            const result = await apiCall('/alerts/create', {
                method: 'POST',
                body: JSON.stringify(formData)
            });
            
            if (result.status === 'SUCCESS') {
                showNotification('هشدار با موفقیت ایجاد شد', 'success');
                form.reset();
                loadAlerts(); // بارگذاری مجدد لیست
            }
        } catch (error) {
            handleError(error, 'ایجاد هشدار');
        }
    });
}

// حذف هشدار
async function deleteAlert(alertId) {
    if (!confirm('آیا از حذف این هشدار مطمئن هستید؟')) {
        return;
    }
    
    try {
        const result = await apiCall(`/alerts/${alertId}`, {
            method: 'DELETE'
        });
        
        if (result.status === 'SUCCESS') {
            showNotification('هشدار با موفقیت حذف شد', 'success');
            loadAlerts(); // بارگذاری مجدد لیست
        }
    } catch (error) {
        handleError(error, 'حذف هشدار');
    }
}

// بارگذاری تاریخچه هشدارها
async function loadAlertHistory() {
    try {
        // در این نسخه، از localStorage استفاده می‌کنیم
        const history = JSON.parse(localStorage.getItem('alertHistory') || '[]');
        alertHistory = history;
        renderAlertHistory();
    } catch (error) {
        console.error('Error loading alert history:', error);
    }
}

// رندر تاریخچه هشدارها
function renderAlertHistory() {
    const container = document.getElementById('alerts-history');
    
    if (!alertHistory || alertHistory.length === 0) {
        container.innerHTML = `
            <div class="text-center py-4 text-gray-500">
                <p>تاریخچه‌ای یافت نشد</p>
            </div>
        `;
        return;
    }
    
    container.innerHTML = alertHistory.slice(0, 10).map(alert => `
        <div class="border-b border-gray-200 py-3 last:border-b-0">
            <div class="flex justify-between items-center">
                <div>
                    <span class="font-semibold">${alert.symbol}</span>
                    <span class="text-sm text-gray-600 mr-2">${alert.message}</span>
                </div>
                <div class="text-sm text-gray-500">
                    ${formatDate(alert.timestamp)}
                </div>
            </div>
        </div>
    `).join('');
}

// utility functions
function formatDate(dateString) {
    try {
        const date = new Date(dateString);
        return new Intl.DateTimeFormat('fa-IR', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        }).format(date);
    } catch (error) {
        return dateString;
    }
}
