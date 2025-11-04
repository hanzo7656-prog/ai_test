// سیستم مدیریت کش هوشمند
class CacheManager {
    constructor() {
        this.memoryCache = new Map();
        this.cleanupInterval = setInterval(() => this.cleanup(), 60000); // هر 1 دقیقه
    }

    set(key, data, ttl = 5 * 60 * 1000) { // پیش‌فرض 5 دقیقه
        const item = {
            data: data,
            expiry: Date.now() + ttl,
            timestamp: Date.now()
        };

        // کش حافظه
        this.memoryCache.set(key, item);

        // کش localStorage برای داده‌های مهم
        if (ttl > 60000) { // فقط برای TTL بیشتر از 1 دقیقه
            try {
                localStorage.setItem(key, JSON.stringify(item));
            } catch (e) {
                console.warn('خطا در ذخیره localStorage:', e);
            }
        }
    }

    get(key) {
        // اول از حافظه بررسی کن
        let item = this.memoryCache.get(key);
        
        if (!item) {
            // سپس از localStorage بررسی کن
            try {
                const stored = localStorage.getItem(key);
                if (stored) {
                    item = JSON.parse(stored);
                    this.memoryCache.set(key, item); // به حافظه برگردون
                }
            } catch (e) {
                console.warn('خطا در خواندن localStorage:', e);
            }
        }

        if (!item) return null;

        // بررسی انقضا
        if (Date.now() > item.expiry) {
            this.delete(key);
            return null;
        }

        return item.data;
    }

    delete(key) {
        this.memoryCache.delete(key);
        try {
            localStorage.removeItem(key);
        } catch (e) {
            console.warn('خطا در حذف از localStorage:', e);
        }
    }

    cleanup() {
        const now = Date.now();
        
        // پاکسازی حافظه
        for (const [key, item] of this.memoryCache.entries()) {
            if (now > item.expiry) {
                this.memoryCache.delete(key);
            }
        }

        // پاکسازی localStorage (هر 5 دقیقه)
        if (Math.random() < 0.2) { // 20% شانس در هر اجرا
            this.cleanupLocalStorage();
        }
    }

    cleanupLocalStorage() {
        try {
            const keysToRemove = [];
            for (let i = 0; i < localStorage.length; i++) {
                const key = localStorage.key(i);
                if (key && key.startsWith('scan_')) {
                    try {
                        const item = JSON.parse(localStorage.getItem(key));
                        if (item && Date.now() > item.expiry) {
                            keysToRemove.push(key);
                        }
                    } catch (e) {
                        keysToRemove.push(key); // داده خراب
                    }
                }
            }

            keysToRemove.forEach(key => localStorage.removeItem(key));
        } catch (e) {
            console.warn('خطا در پاکسازی localStorage:', e);
        }
    }

    // پاکسازی کامل کش
    clear() {
        this.memoryCache.clear();
        try {
            for (let i = 0; i < localStorage.length; i++) {
                const key = localStorage.key(i);
                if (key && key.startsWith('scan_')) {
                    localStorage.removeItem(key);
                }
            }
        } catch (e) {
            console.warn('خطا در پاکسازی کامل کش:', e);
        }
    }

    // آمار کش
    getStats() {
        let memorySize = 0;
        let localStorageSize = 0;

        // محاسبه سایز حافظه
        for (const item of this.memoryCache.values()) {
            memorySize += JSON.stringify(item).length;
        }

        // محاسبه سایز localStorage
        try {
            for (let i = 0; i < localStorage.length; i++) {
                const key = localStorage.key(i);
                if (key && key.startsWith('scan_')) {
                    localStorageSize += localStorage.getItem(key).length;
                }
            }
        } catch (e) {}

        return {
            memory: {
                count: this.memoryCache.size,
                size: Math.round(memorySize / 1024) + ' KB'
            },
            localStorage: {
                count: this.getLocalStorageCount(),
                size: Math.round(localStorageSize / 1024) + ' KB'
            }
        };
    }

    getLocalStorageCount() {
        let count = 0;
        try {
            for (let i = 0; i < localStorage.length; i++) {
                const key = localStorage.key(i);
                if (key && key.startsWith('scan_')) {
                    count++;
                }
            }
        } catch (e) {}
        return count;
    }
}

// نمونه جهانی
const cacheManager = new CacheManager();
