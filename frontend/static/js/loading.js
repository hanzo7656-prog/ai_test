// سیستم لودینگ هوشمند - سازگار با HTML موجود
class SmartLoading {
    constructor() {
        this.isVisible = false;
        this.startTime = null;
        this.intervalId = null;
    }

    start(options) {
        this.isVisible = true;
        this.startTime = Date.now();
        this.options = options;
        
        this.show();
        this.startTimer();
        
        // آپدیت عنوان
        const titleEl = document.getElementById('loadingTitle');
        if (titleEl) {
            titleEl.textContent = `در حال اسکن ${options.total} ارز - حالت ${options.isAIMode ? 'AI' : 'ساده'}`;
        }
    }

    updateProgress(completed, total, currentBatch = []) {
        if (!this.isVisible) return;

        const percent = Math.round((completed / total) * 100);
        const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
        const speed = elapsed > 0 ? Math.round((completed / elapsed) * 60) : 0;

        // ✅ استفاده از IDهای موجود در HTML
        const progressFill = document.getElementById('progressFill');
        if (progressFill) progressFill.style.width = percent + '%';
        
        const progressPercent = document.getElementById('progressPercent');
        if (progressPercent) progressPercent.textContent = percent + '%';
        
        const progressText = document.getElementById('progressText');
        if (progressText) progressText.textContent = `${completed}/${total}`;
        
        const elapsedTime = document.getElementById('elapsedTime');
        if (elapsedTime) elapsedTime.textContent = this.formatTime(elapsed);
        
        const scanSpeed = document.getElementById('scanSpeed');
        if (scanSpeed) scanSpeed.textContent = `${speed}/دقیقه`;

        // نمایش ارزهای در حال اسکن
        this.updateScanningList(currentBatch);
    }

    updateScanningList(currentBatch) {
        const scanningList = document.getElementById('scanningList');
        if (scanningList && currentBatch.length > 0) {
            const limitedSymbols = currentBatch.slice(0, 5);
            scanningList.innerHTML = limitedSymbols
                .map(symbol => `<span class="coin-tag scanning">${symbol.toUpperCase()}</span>`)
                .join('');
        }
    }

    startTimer() {
        this.intervalId = setInterval(() => {
            if (this.isVisible) {
                const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
                const elapsedTime = document.getElementById('elapsedTime');
                if (elapsedTime) elapsedTime.textContent = this.formatTime(elapsed);
            }
        }, 1000);
    }

    formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    }

    show() {
        const loadingOverlay = document.getElementById('loadingOverlay');
        if (loadingOverlay) {
            loadingOverlay.style.display = 'flex';
            setTimeout(() => {
                loadingOverlay.style.opacity = '1';
            }, 10);
        }
    }

    hide() {
        const loadingOverlay = document.getElementById('loadingOverlay');
        if (loadingOverlay) {
            loadingOverlay.style.opacity = '0';
            setTimeout(() => {
                loadingOverlay.style.display = 'none';
            }, 300);
        }
    }

    complete() {
        this.isVisible = false;
        clearInterval(this.intervalId);
        this.hide();
    }

    showError(message) {
        this.complete();
        // ✅ استفاده از سیستم نوتیفیکیشن موجود به جای alert
        if (window.vortexApp && window.vortexApp.uiManager) {
            window.vortexApp.uiManager.showNotification('خطا: ' + message, 'error');
        } else {
            console.error('خطا:', message);
        }
    }
}

// نمونه جهانی
const smartLoading = new SmartLoading();
