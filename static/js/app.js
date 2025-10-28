// متغیرهای全局
let websocketConnected = false;
let currentCharts = {};

// مقداردهی اولیه
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    startWebSocketConnection();
    updateCurrentTime();
});

// راه‌اندازی برنامه
function initializeApp() {
    console.log('🤖 AI Trading Assistant Initialized');
    
    // بروزرسانی زمان جاری
    setInterval(updateCurrentTime, 1000);
    
    // بارگذاری داده‌های اولیه
    loadInitialData();
}

// بروزرسانی زمان
function updateCurrentTime() {
    const now = new Date();
    const timeString = now.toLocaleTimeString('fa-IR');
    const dateString = now.toLocaleDateString('fa-IR');
    
    const timeElement = document.getElementById('current-time');
    if (timeElement) {
        timeElement.textContent = `${dateString} - ${timeString}`;
    }
}

// بارگذاری داده‌های اولیه
async function loadInitialData() {
    try {
        // وضعیت WebSocket
        const wsStatus = await fetch('/websocket/status');
        const wsData = await wsStatus.json();
        updateWebSocketStatus(wsData);
        
        // شاخص ترس و طمع
        const fearGreed = await fetch('/insights/fear-greed');
        const fgData = await fearGreed.json();
        updateFearGreedIndex(fgData);
        
        // دامیننس بیت‌کوین
        const btcDom = await fetch('/insights/btc-dominance');
        const btcData = await btcDom.json();
        updateBTCDominance(btcData);
        
        // کوین‌های برتر
        const topCoins = await fetch('/coins/list?limit=10');
        const coinsData = await topCoins.json();
        updateTopCoins(coinsData);
        
    } catch (error) {
        console.error('Error loading initial data:', error);
    }
}

// بروزرسانی وضعیت WebSocket
function updateWebSocketStatus(data) {
    const statusElement = document.getElementById('websocket-status');
    const pairsElement = document.getElementById('active-pairs');
    const connectionElement = document.getElementById('connection-status');
    
    if (statusElement && pairsElement && connectionElement) {
        if (data.connected) {
            statusElement.textContent = 'متصل';
            statusElement.className = 'text-2xl font-bold text-green-600';
            pairsElement.textContent = `${data.active_pairs.length} جفت فعال`;
            
            connectionElement.innerHTML = `
                <div class="w-3 h-3 rounded-full bg-green-500 mr-2"></div>
                <span class="text-sm">متصل</span>
            `;
            
            websocketConnected = true;
        } else {
            statusElement.textContent = 'قطع';
            statusElement.className = 'text-2xl font-bold text-red-600';
            pairsElement.textContent = 'اتصال قطع است';
            
            connectionElement.innerHTML = `
                <div class="w-3 h-3 rounded-full bg-red-500 mr-2"></div>
                <span class="text-sm">قطع</span>
            `;
            
            websocketConnected = false;
        }
    }
}

// بروزرسانی شاخص ترس و طمع
function updateFearGreedIndex(data) {
    const valueElement = document.getElementById('fear-greed-value');
    const textElement = document.getElementById('fear-greed-text');
    
    if (valueElement && textElement && data.now) {
        const value = data.now.value;
        valueElement.textContent = value;
        
        let status = 'خنثی';
        let color = 'text-yellow-600';
        
        if (value >= 70) {
            status = 'طمع شدید';
            color = 'text-red-600';
        } else if (value >= 55) {
            status = 'طمع';
            color = 'text-yellow-600';
        } else if (value >= 45) {
            status = 'خنثی';
            color = 'text-gray-600';
        } else if (value >= 30) {
            status = 'ترس';
            color = 'text-blue-600';
        } else {
            status = 'ترس شدید';
            color = 'text-green-600';
        }
        
        valueElement.className = `text-2xl font-bold ${color}`;
        textElement.textContent = `حالت: ${status}`;
    }
}

// نمایش نوتیفیکیشن
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 ${
        type === 'success' ? 'bg-green-500 text-white' :
        type === 'error' ? 'bg-red-500 text-white' :
        'bg-blue-500 text-white'
    }`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.remove();
    }, 5000);
}

// مدیریت خطا
function handleError(error, context = 'عملیات') {
    console.error(`${context} error:`, error);
    showNotification(`خطا در ${context}: ${error.message}`, 'error');
}

// API Calls
async function apiCall(endpoint, options = {}) {
    try {
        const response = await fetch(endpoint, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        handleError(error, `API call to ${endpoint}`);
        throw error;
    }
}
