// ابزارهای کمکی و توابع عمومی VortexAI
class VortexUtils {
    constructor() {
        this.cache = new Map();
    }

    // فرمت‌دهی قیمت
    static formatPrice(price) {
        if (price === 0 || price === null || price === undefined) return '0.00';
        if (price < 0.01) return price.toFixed(6);
        if (price < 1) return price.toFixed(4);
        if (price < 1000) return price.toFixed(2);
        return price.toLocaleString('en-US', { maximumFractionDigits: 2 });
    }

    // فرمت‌دهی اعداد بزرگ
    static formatNumber(num) {
        if (num === 0 || num === null || num === undefined) return '0';
        if (num < 1000) return num.toString();
        if (num < 1000000) return (num / 1000).toFixed(1) + 'K';
        if (num < 1000000000) return (num / 1000000).toFixed(1) + 'M';
        if (num < 1000000000000) return (num / 1000000000).toFixed(1) + 'B';
        return (num / 1000000000000).toFixed(1) + 'T';
    }

    // محاسبه تازگی داده
    static getDataFreshness(timestamp) {
        if (!timestamp) return 'نامشخص';
        
        try {
            const now = new Date();
            const dataTime = new Date(timestamp);
            
            if (isNaN(dataTime.getTime())) return 'نامشخص';
            
            const diffMinutes = Math.round((now - dataTime) / (1000 * 60));
            
            if (diffMinutes < 1) return 'همین لحظه';
            if (diffMinutes < 5) return 'دقایقی پیش';
            if (diffMinutes < 30) return 'اخیراً';
            if (diffMinutes < 60) return 'کمتر از 1 ساعت';
            if (diffMinutes < 120) return '1 ساعت پیش';
            return 'قدیمی';
        } catch {
            return 'نامشخص';
        }
    }

    // فرمت‌دهی زمان
    static formatTime(seconds) {
        if (!seconds) return '0:00';
        
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    }

    // Escape HTML
    static escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // ایجاد هش از رشته
    static stringToHash(str) {
        if (!str) return 0;
        
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;
        }
        return Math.abs(hash);
    }

    // دانلود فایل
    static downloadFile(filename, content) {
        try {
            const blob = new Blob([content], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            return true;
        } catch (error) {
            console.error('Download error:', error);
            return false;
        }
    }

    // تاخیر
    static delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // نماد ارزها
    static getCoinSymbol(symbol) {
        if (!symbol) return '???';
        
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
            'bitcoin-cash': 'BCH',
            'stellar': 'XLM',
            'monero': 'XMR',
            'ethereum-classic': 'ETC',
            'vechain': 'VET',
            'theta-token': 'THETA'
        };
        return symbolsMap[symbol.toLowerCase()] || symbol.substring(0, 3).toUpperCase();
    }

    // اعتبارسنجی سمبل
    static isValidSymbol(symbol) {
        if (!symbol || typeof symbol !== 'string') return false;
        return /^[a-zA-Z0-9-]+$/.test(symbol) && symbol.length >= 2 && symbol.length <= 20;
    }

    // بررسی وجود المان
    static getElement(id) {
        const element = document.getElementById(id);
        if (!element) {
            console.warn(`Element with id '${id}' not found`);
        }
        return element;
    }

    // نمایش خطا در console با استایل
    static logError(message, data = null) {
        console.error(`%c❌ ${message}`, 'color: #ff4757; font-weight: bold;');
        if (data) console.error(data);
    }

    static logSuccess(message, data = null) {
        console.log(`%c✅ ${message}`, 'color: #00d9a6; font-weight: bold;');
        if (data) console.log(data);
    }

    static logInfo(message, data = null) {
        console.log(`%cℹ️ ${message}`, 'color: #0052ff; font-weight: bold;');
        if (data) console.log(data);
    }

    // مدیریت کش ساده
    setCache(key, value, ttl = 300000) { // 5 دقیقه پیش‌فرض
        this.cache.set(key, {
            value,
            expiry: Date.now() + ttl
        });
    }

    getCache(key) {
        const item = this.cache.get(key);
        if (!item) return null;
        
        if (Date.now() > item.expiry) {
            this.cache.delete(key);
            return null;
        }
        
        return item.value;
    }

    clearCache() {
        this.cache.clear();
    }

    // تولید رنگ بر اساس سمبل
    static getSymbolColor(symbol) {
        const colors = [
            '#ff6b6b', '#51cf66', '#339af0', '#ff922b', '#cc5de8',
            '#20c997', '#f06595', '#748ffc', '#63e6be', '#ffd43b'
        ];
        const hash = this.stringToHash(symbol);
        return colors[hash % colors.length];
    }

    // فرمت تاریخ فارسی
    static formatPersianDate(date) {
        try {
            const options = { 
                year: 'numeric', 
                month: 'long', 
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            };
            return new Date(date).toLocaleDateString('fa-IR', options);
        } catch {
            return new Date().toLocaleDateString('fa-IR');
        }
    }

    // بررسی آنلاین بودن
    static isOnline() {
        return navigator.onLine;
    }

    // کپی به کلیپ‌بورد
    static async copyToClipboard(text) {
        try {
            await navigator.clipboard.writeText(text);
            return true;
        } catch {
            // Fallback برای مرورگرهای قدیمی
            const textArea = document.createElement('textarea');
            textArea.value = text;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            return true;
        }
    }

    // اندازه‌گیری عملکرد
    static measurePerformance(name, fn) {
        const start = performance.now();
        const result = fn();
        const end = performance.now();
        console.log(`⏱️ ${name}: ${(end - start).toFixed(2)}ms`);
        return result;
    }

    // تولید ID یکتا
    static generateId(prefix = '') {
        return `${prefix}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }
}

// ایجاد نمونه جهانی
window.VortexUtils = VortexUtils;
window.vortexUtils = new VortexUtils();

// توابع قدیمی برای سازگاری
window.formatPrice = VortexUtils.formatPrice;
window.formatNumber = VortexUtils.formatNumber;
window.getDataFreshness = VortexUtils.getDataFreshness;
window.formatTime = VortexUtils.formatTime;
window.escapeHtml = VortexUtils.escapeHtml;
window.stringToHash = VortexUtils.stringToHash;
window.downloadFile = VortexUtils.downloadFile;
window.delay = VortexUtils.delay;
window.getCoinSymbol = VortexUtils.getCoinSymbol;
