// static/js/settings.js
class UserSettings {
    constructor() {
        this.currentTab = 'profile';
        this.settings = this.loadSettings();
        this.initializeTabs();
        this.setupEventListeners();
        this.loadCurrentSettings();
    }

    initializeTabs() {
        // فعال‌سازی تب پیش‌فرض
        this.showTab('profile');
    }

    setupEventListeners() {
        // مدیریت کلیک روی تب‌ها
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const tab = e.currentTarget.dataset.tab;
                this.showTab(tab);
            });
        });

        // ذخیره تنظیمات پروفایل
        document.getElementById('saveProfile')?.addEventListener('click', () => {
            this.saveProfileSettings();
        });

        document.getElementById('resetProfile')?.addEventListener('click', () => {
            this.resetProfileSettings();
        });

        // مدیریت تم‌ها
        document.querySelectorAll('.theme-option').forEach(option => {
            option.addEventListener('click', (e) => {
                this.selectTheme(e.currentTarget.dataset.theme);
            });
        });

        // مدیریت اسلایدرها
        document.getElementById('alertThreshold')?.addEventListener('input', (e) => {
            this.updateSliderValue('alertThresholdValue', e.target.value, '%');
        });

        document.getElementById('signalConfidence')?.addEventListener('input', (e) => {
            this.updateSliderValue('signalConfidenceValue', e.target.value, '%');
        });

        // ذخیره خودکار تنظیمات
        this.setupAutoSave();
    }

    showTab(tabName) {
        // غیرفعال کردن همه تب‌ها
        document.querySelectorAll('.tab-content').forEach(tab => {
            tab.classList.remove('active');
        });
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });

        // فعال کردن تب انتخاب شده
        const targetTab = document.getElementById(`${tabName}-tab`);
        const targetNav = document.querySelector(`[data-tab="${tabName}"]`);
        
        if (targetTab && targetNav) {
            targetTab.classList.add('active');
            targetNav.classList.add('active');
            this.currentTab = tabName;
        }
    }

    loadSettings() {
        // بارگذاری تنظیمات از localStorage
        const saved = localStorage.getItem('vortexai-settings');
        if (saved) {
            return JSON.parse(saved);
        }

        // تنظیمات پیش‌فرض
        return {
            profile: {
                username: 'VortexTrader',
                email: 'trader@example.com',
                phone: '+98 912 345 6789',
                language: 'fa',
                timezone: 'tehran',
                currency: 'usdt'
            },
            appearance: {
                theme: 'dark-blue',
                chartStyle: 'candle',
                bullishColor: '#10b981',
                bearishColor: '#ef4444',
                dataDensity: 'normal',
                animations: true,
                fontFamily: 'vazirmatn'
            },
            notifications: {
                priceAlerts: true,
                alertThreshold: 3,
                aiSignals: true,
                signalConfidence: 75,
                browserNotifications: true,
                emailNotifications: false,
                soundNotifications: false
            },
            trading: {
                // تنظیمات معاملاتی
            },
            aiModels: {
                // تنظیمات مدل‌های AI
            }
        };
    }

    saveSettings() {
        // ذخیره تنظیمات در localStorage
        localStorage.setItem('vortexai-settings', JSON.stringify(this.settings));
        this.showNotification('تنظیمات ذخیره شد');
    }

    loadCurrentSettings() {
        // بارگذاری تنظیمات فعلی در فرم
        this.loadProfileSettings();
        this.loadAppearanceSettings();
        this.loadNotificationSettings();
    }

    loadProfileSettings() {
        const profile = this.settings.profile;
        
        document.getElementById('username').value = profile.username;
        document.getElementById('email').value = profile.email;
        document.getElementById('phone').value = profile.phone;
        document.getElementById('language').value = profile.language;
        document.getElementById('timezone').value = profile.timezone;
        document.getElementById('currency').value = profile.currency;
    }

    loadAppearanceSettings() {
        const appearance = this.settings.appearance;
        
        // فعال کردن تم انتخاب شده
        document.querySelectorAll('.theme-option').forEach(option => {
            option.classList.remove('active');
        });
        document.querySelector(`[data-theme="${appearance.theme}"]`)?.classList.add('active');
        
        document.getElementById('chartStyle').value = appearance.chartStyle;
        document.getElementById('bullishColor').value = appearance.bullishColor;
        document.getElementById('bearishColor').value = appearance.bearishColor;
        document.getElementById('dataDensity').value = appearance.dataDensity;
        document.getElementById('animationsToggle').checked = appearance.animations;
        document.getElementById('fontFamily').value = appearance.fontFamily;
    }

    loadNotificationSettings() {
        const notifications = this.settings.notifications;
        
        document.getElementById('priceAlertsToggle').checked = notifications.priceAlerts;
        document.getElementById('alertThreshold').value = notifications.alertThreshold;
        document.getElementById('alertThresholdValue').textContent = `${notifications.alertThreshold}%`;
        
        document.getElementById('aiSignalsToggle').checked = notifications.aiSignals;
        document.getElementById('signalConfidence').value = notifications.signalConfidence;
        document.getElementById('signalConfidenceValue').textContent = `${notifications.signalConfidence}%`;
        
        document.getElementById('browserNotificationsToggle').checked = notifications.browserNotifications;
        document.getElementById('emailNotificationsToggle').checked = notifications.emailNotifications;
        document.getElementById('soundNotificationsToggle').checked = notifications.soundNotifications;
    }

    saveProfileSettings() {
        this.settings.profile = {
            username: document.getElementById('username').value,
            email: document.getElementById('email').value,
            phone: document.getElementById('phone').value,
            language: document.getElementById('language').value,
            timezone: document.getElementById('timezone').value,
            currency: document.getElementById('currency').value
        };

        this.saveSettings();
    }

    resetProfileSettings() {
        if (confirm('آیا از بازنشانی تنظیمات پروفایل اطمینان دارید؟')) {
            this.settings.profile = {
                username: 'VortexTrader',
                email: 'trader@example.com',
                phone: '+98 912 345 6789',
                language: 'fa',
                timezone: 'tehran',
                currency: 'usdt'
            };
            
            this.loadProfileSettings();
            this.saveSettings();
        }
    }

    selectTheme(theme) {
        this.settings.appearance.theme = theme;
        
        // آپدیت UI
        document.querySelectorAll('.theme-option').forEach(option => {
            option.classList.remove('active');
        });
        document.querySelector(`[data-theme="${theme}"]`)?.classList.add('active');
        
        // اعمال تم (در واقعیت باید CSS رو عوض کنه)
        this.applyTheme(theme);
        this.saveSettings();
    }

    applyTheme(theme) {
        // اینجا می‌تونید تم رو روی صفحه اعمال کنید
        console.log(`تم ${theme} اعمال شد`);
        
        // برای نمونه، تغییر کلاس body
        document.body.className = '';
        document.body.classList.add(theme);
    }

    updateSliderValue(elementId, value, suffix = '') {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = `${value}${suffix}`;
        }
    }

    setupAutoSave() {
        // ذخیره خودکار هنگام تغییر تنظیمات
        const autoSaveElements = [
            'chartStyle', 'dataDensity', 'fontFamily', 'language', 
            'timezone', 'currency'
        ];

        autoSaveElements.forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.addEventListener('change', () => {
                    this.saveCurrentTabSettings();
                });
            }
        });

        // برای toggleها
        const toggleElements = [
            'animationsToggle', 'priceAlertsToggle', 'aiSignalsToggle',
            'browserNotificationsToggle', 'emailNotificationsToggle', 'soundNotificationsToggle'
        ];

        toggleElements.forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.addEventListener('change', () => {
                    this.saveCurrentTabSettings();
                });
            }
        });

        // برای اسلایدرها
        const sliderElements = ['alertThreshold', 'signalConfidence'];
        sliderElements.forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.addEventListener('change', () => {
                    this.saveCurrentTabSettings();
                });
            }
        });

        // برای رنگ‌ها
        const colorElements = ['bullishColor', 'bearishColor'];
        colorElements.forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.addEventListener('change', () => {
                    this.saveCurrentTabSettings();
                });
            }
        });
    }

    saveCurrentTabSettings() {
        switch (this.currentTab) {
            case 'profile':
                this.saveProfileSettings();
                break;
            case 'appearance':
                this.saveAppearanceSettings();
                break;
            case 'notifications':
                this.saveNotificationSettings();
                break;
            case 'trading':
                this.saveTradingSettings();
                break;
            case 'ai-models':
                this.saveAIModelSettings();
                break;
        }
    }

    saveAppearanceSettings() {
        this.settings.appearance = {
            theme: this.settings.appearance.theme, // حفظ تم فعلی
            chartStyle: document.getElementById('chartStyle').value,
            bullishColor: document.getElementById('bullishColor').value,
            bearishColor: document.getElementById('bearishColor').value,
            dataDensity: document.getElementById('dataDensity').value,
            animations: document.getElementById('animationsToggle').checked,
            fontFamily: document.getElementById('fontFamily').value
        };

        this.saveSettings();
    }

    saveNotificationSettings() {
        this.settings.notifications = {
            priceAlerts: document.getElementById('priceAlertsToggle').checked,
            alertThreshold: parseInt(document.getElementById('alertThreshold').value),
            aiSignals: document.getElementById('aiSignalsToggle').checked,
            signalConfidence: parseInt(document.getElementById('signalConfidence').value),
            browserNotifications: document.getElementById('browserNotificationsToggle').checked,
            emailNotifications: document.getElementById('emailNotificationsToggle').checked,
            soundNotifications: document.getElementById('soundNotificationsToggle').checked
        };

        this.saveSettings();
    }

    saveTradingSettings() {
        // ذخیره تنظیمات معاملاتی
        console.log('تنظیمات معاملاتی ذخیره شد');
    }

    saveAIModelSettings() {
        // ذخیره تنظیمات مدل‌های AI
        console.log('تنظیمات AI ذخیره شد');
    }

    showNotification(message) {
        // ایجاد نوتیفیکیشن
        const notification = document.createElement('div');
        notification.className = 'settings-notification';
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: var(--accent-success);
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            z-index: 10000;
            animation: slideInRight 0.3s ease;
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease';
            setTimeout(() => {
                notification.remove();
            }, 300);
        }, 3000);
    }

    exportSettings() {
        // صادر کردن تنظیمات
        const dataStr = JSON.stringify(this.settings, null, 2);
        const dataBlob = new Blob([dataStr], {type: 'application/json'});
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = 'vortexai-settings.json';
        link.click();
    }

    importSettings(event) {
        // وارد کردن تنظیمات
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const importedSettings = JSON.parse(e.target.result);
                    this.settings = {...this.settings, ...importedSettings};
                    this.loadCurrentSettings();
                    this.saveSettings();
                    this.showNotification('تنظیمات با موفقیت وارد شد');
                } catch (error) {
                    this.showNotification('خطا در وارد کردن تنظیمات');
                }
            };
            reader.readAsText(file);
        }
    }
}

// راه‌اندازی
document.addEventListener('DOMContentLoaded', () => {
    new UserSettings();
});
