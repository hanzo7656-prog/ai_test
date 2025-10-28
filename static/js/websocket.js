// مدیریت اتصال WebSocket
class WebSocketManager {
    constructor() {
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 3000;
        this.isConnected = false;
        this.subscribedPairs = new Set();
    }
    
    // اتصال به WebSocket
    connect() {
        try {
            this.ws = new WebSocket('wss://api.lbank.com/ws/V2/');
            
            this.ws.onopen = () => {
                console.log('✅ WebSocket connected');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.updateConnectionStatus(true);
                this.resubscribeAll();
            };
            
            this.ws.onmessage = (event) => {
                this.handleMessage(event.data);
            };
            
            this.ws.onclose = () => {
                console.log('❌ WebSocket disconnected');
                this.isConnected = false;
                this.updateConnectionStatus(false);
                this.handleReconnection();
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.isConnected = false;
                this.updateConnectionStatus(false);
            };
            
        } catch (error) {
            console.error('Error connecting to WebSocket:', error);
            this.handleReconnection();
        }
    }
    
    // مدیریت پیام‌های دریافتی
    handleMessage(data) {
        try {
            const message = JSON.parse(data);
            
            if (message.action === 'update' && message.data) {
                this.updatePriceData(message.data);
            } else if (message.ping) {
                // پاسخ به ping
                this.ws.send(JSON.stringify({ pong: message.ping }));
            }
            
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    }
    
    // به‌روزرسانی داده‌های قیمت
    updatePriceData(data) {
        // به‌روزرسانی قیمت در رابط کاربری
        if (data.symbol && data.price) {
            this.updateUIPrice(data.symbol, data.price);
            
            // به‌روزرسانی نمودارها
            if (charts[data.symbol.toLowerCase()]) {
                updateRealtimeChart(data.symbol, data.price);
            }
            
            // بررسی هشدارها
            this.checkAlerts(data.symbol, data.price);
        }
    }
    
    // به‌روزرسانی قیمت در UI
    updateUIPrice(symbol, price) {
        // به‌روزرسانی در صفحه بازار
        const priceElement = document.getElementById(`ws-price-${symbol}`);
        if (priceElement) {
            priceElement.textContent = `$${formatNumber(price)}`;
        }
        
        // به‌روزرسانی در صفحه جزئیات
        const detailPriceElement = document.getElementById(`current-price-${symbol}`);
        if (detailPriceElement) {
            detailPriceElement.textContent = `$${formatNumber(price)}`;
        }
    }
    
    // بررسی هشدارها
    checkAlerts(symbol, price) {
        const relevantAlerts = alerts.filter(alert => 
            alert.symbol === symbol && alert.status === 'ACTIVE'
        );
        
        relevantAlerts.forEach(alert => {
            if (this.shouldTriggerAlert(alert, price)) {
                this.triggerAlert(alert, price);
            }
        });
    }
    
    // بررسی فعال‌سازی هشدار
    shouldTriggerAlert(alert, currentPrice) {
        switch (alert.condition) {
            case 'above':
                return currentPrice >= alert.target_price;
            case 'below':
                return currentPrice <= alert.target_price;
            default:
                return false;
        }
    }
    
    // فعال‌سازی هشدار
    triggerAlert(alert, currentPrice) {
        showNotification(
            `🚨 هشدار ${alert.symbol}: قیمت به $${formatNumber(currentPrice)} رسید!`,
            'success'
        );
        
        // اضافه کردن به تاریخچه
        this.addToAlertHistory(alert, currentPrice);
        
        // غیرفعال کردن هشدار
        this.deactivateAlert(alert.id);
    }
    
    // اضافه کردن به تاریخچه هشدارها
    addToAlertHistory(alert, triggeredPrice) {
        const historyItem = {
            symbol: alert.symbol,
            message: `هشدار ${alert.condition === 'above' ? 'بالاتر از' : 'پایین‌تر از'} $${alert.target_price} فعال شد`,
            triggeredPrice: triggeredPrice,
            targetPrice: alert.target_price,
            timestamp: new Date().toISOString()
        };
        
        alertHistory.unshift(historyItem);
        localStorage.setItem('alertHistory', JSON.stringify(alertHistory));
        renderAlertHistory();
    }
    
    // غیرفعال کردن هشدار
    deactivateAlert(alertId) {
        const alertIndex = alerts.findIndex(alert => alert.id === alertId);
        if (alertIndex !== -1) {
            alerts[alertIndex].status = 'TRIGGERED';
            renderAlertsList();
        }
    }
    
    // عضویت در جفت ارز
    subscribe(pair) {
        if (!this.isConnected || !this.ws) {
            console.warn('WebSocket not connected, queueing subscription:', pair);
            this.subscribedPairs.add(pair);
            return;
        }
        
        const subscribeMessage = {
            action: 'subscribe',
            subscribe: `tick_${pair}`
        };
        
        this.ws.send(JSON.stringify(subscribeMessage));
        this.subscribedPairs.add(pair);
        console.log(`✅ Subscribed to: ${pair}`);
    }
    
    // لغو عضویت
    unsubscribe(pair) {
        if (!this.isConnected || !this.ws) return;
        
        const unsubscribeMessage = {
            action: 'unsubscribe',
            unsubscribe: `tick_${pair}`
        };
        
        this.ws.send(JSON.stringify(unsubscribeMessage));
        this.subscribedPairs.delete(pair);
        console.log(`❌ Unsubscribed from: ${pair}`);
    }
    
    // عضویت مجدد همه
    resubscribeAll() {
        this.subscribedPairs.forEach(pair => {
            this.subscribe(pair);
        });
    }
    
    // مدیریت اتصال مجدد
    handleReconnection() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('Max reconnection attempts reached');
            return;
        }
        
        this.reconnectAttempts++;
        const delay = this.reconnectDelay * this.reconnectAttempts;
        
        console.log(`Reconnecting in ${delay}ms... (attempt ${this.reconnectAttempts})`);
        
        setTimeout(() => {
            this.connect();
        }, delay);
    }
    
    // قطع اتصال
    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        this.isConnected = false;
        this.updateConnectionStatus(false);
    }
    
    // به‌روزرسانی وضعیت اتصال در UI
    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('websocket-status');
        const connectionElement = document.getElementById('connection-status');
        
        if (statusElement) {
            statusElement.textContent = connected ? 'متصل' : 'قطع';
            statusElement.className = connected ? 
                'text-2xl font-bold text-green-600' : 
                'text-2xl font-bold text-red-600';
        }
        
        if (connectionElement) {
            connectionElement.innerHTML = connected ? 
                '<div class="w-3 h-3 rounded-full bg-green-500 mr-2"></div><span class="text-sm">متصل</span>' :
                '<div class="w-3 h-3 rounded-full bg-red-500 mr-2"></div><span class="text-sm">قطع</span>';
        }
        
        // به‌روزرسانی در سایر صفحات
        const globalStatusElement = document.getElementById('global-websocket-status');
        if (globalStatusElement) {
            globalStatusElement.textContent = connected ? 'متصل' : 'قطع';
            globalStatusElement.className = connected ? 
                'text-green-600 font-semibold' : 
                'text-red-600 font-semibold';
        }
    }
    
    // دریافت داده‌های لحظه‌ای
    getRealtimeData(symbol = null) {
        // این تابع می‌تواند داده‌های کش شده را برگرداند
        // در یک پیاده‌سازی واقعی، این داده‌ها از سرور گرفته می‌شوند
        return symbol ? this.priceData[symbol] : this.priceData;
    }
}

// ایجاد نمونه WebSocket Manager
const wsManager = new WebSocketManager();

// شروع اتصال هنگام بارگذاری صفحه
document.addEventListener('DOMContentLoaded', function() {
    // تأخیر کوچک برای اجازه بارگذاری اولیه
    setTimeout(() => {
        wsManager.connect();
        
        // عضویت در جفت ارزهای اصلی
        const majorPairs = ['btc_usdt', 'eth_usdt', 'sol_usdt', 'bnb_usdt'];
        majorPairs.forEach(pair => {
            wsManager.subscribe(pair);
        });
    }, 2000);
});
