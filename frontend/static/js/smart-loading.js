// سیستم لودینگ هوشمند
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
        
        this.updateDOM();
        this.show();
        this.startTimer();
    }

    updateProgress(completed, total, currentScanning = [], completedScan = []) {
        if (!this.isVisible) return;

        const percent = Math.round((completed / total) * 100);
        const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
        const remaining = completed > 0 ? 
            Math.floor((elapsed / completed) * (total - completed)) : 0;
        
        const speed = elapsed > 0 ? Math.round((completed / elapsed) * 60) : 0;

        // آپدیت DOM
        document.getElementById('smartProgressBar').style.width = percent + '%';
        document.getElementById('progressPercent').textContent = percent + '%';
        document.getElementById('progressCount').textContent = `${completed}/${total}`;
        
        document.getElementById('elapsedTime').textContent = this.formatTime(elapsed);
        document.getElementById('remainingTime').textContent = this.formatTime(remaining);
        document.getElementById('scanSpeed').textContent = `${speed} ارز/دقیقه`;
        
        this.updateScanningLists(currentScanning, completedScan);
    }

    updateBatchInfo(currentBatch, totalBatches) {
        document.getElementById('batchInfo').textContent = `${currentBatch}/${totalBatches}`;
    }

    updateCurrentScanning(symbols) {
        const limitedSymbols = symbols.slice(0, 5); // فقط 5 تا نمایش بده
        const elements = limitedSymbols.map(symbol => 
            `<span class="coin-tag">${symbol.toUpperCase()}</span>`
        ).join('');
        
        document.getElementById('currentScanning').innerHTML = elements;
    }

    updateScanningLists(currentScanning, completedScan) {
        // نمایش آخرین 3 ارز در حال اسکن
        const currentLimited = currentScanning.slice(-3);
        const currentElements = currentLimited.map(symbol => 
            `<span class="coin-tag scanning">${symbol.toUpperCase()}</span>`
        ).join('');
        
        // نمایش آخرین 5 ارز تکمیل شده
        const completedLimited = completedScan.slice(-5);
        const completedElements = completedLimited.map(symbol => 
            `<span class="coin-tag completed">${symbol.toUpperCase()}</span>`
        ).join('');
        
        document.getElementById('currentScanning').innerHTML = currentElements;
        document.getElementById('completedScan').innerHTML = completedElements;
    }

    updateDOM() {
        document.getElementById('scanModeInfo').textContent = 
            this.options.isAIMode ? 'AI (داده کامل)' : 'Manual (داده بهینه)';
        
        document.getElementById('loadingTitle').textContent = 
            `در حال اسکن ${this.options.scanType} - ${this.options.total} ارز`;
    }

    startTimer() {
        this.intervalId = setInterval(() => {
            if (this.isVisible) {
                const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
                document.getElementById('elapsedTime').textContent = this.formatTime(elapsed);
            }
        }, 1000);
    }

    formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    }

    show() {
        document.getElementById('smartLoading').style.display = 'flex';
        setTimeout(() => {
            document.getElementById('smartLoading').style.opacity = '1';
        }, 10);
    }

    hide() {
        document.getElementById('smartLoading').style.opacity = '0';
        setTimeout(() => {
            document.getElementById('smartLoading').style.display = 'none';
        }, 300);
    }

    complete() {
        this.isVisible = false;
        clearInterval(this.intervalId);
        this.hide();
    }

    showError(message) {
        this.complete();
        alert('خطا: ' + message);
    }
}

// نمونه جهانی
const smartLoading = new SmartLoading();
