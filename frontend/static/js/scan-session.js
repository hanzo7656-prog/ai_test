// سیستم اسکن پیشرفته VortexAI - سازگار با روت‌های جدید
class ScanSession {
    constructor(options) {
        this.symbols = options.symbols;
        this.mode = options.mode; // 'ai' یا 'basic'
        this.batchSize = options.batchSize;
        this.onProgress = options.onProgress;
        this.onComplete = options.onComplete;
        this.onError = options.onError;
        
        this.isCancelled = false;
        this.startTime = null;
        this.completed = 0;
        this.results = [];
    }

    async start() {
        this.startTime = Date.now();
        this.isCancelled = false;
        this.completed = 0;
        this.results = [];

        try {
            const batches = this.createBatches();
            
            for (let i = 0; i < batches.length; i++) {
                if (this.isCancelled) break;

                const batch = batches[i];
                await this.processBatch(batch, i + 1, batches.length);
                
                // تاخیر بین batchها برای کاهش فشار
                if (i < batches.length - 1 && !this.isCancelled) {
                    await this.delay(1000);
                }
            }

            if (!this.isCancelled) {
                this.onComplete?.(this.results);
            }

        } catch (error) {
            this.onError?.(error);
        }
    }

    createBatches() {
        const batches = [];
        for (let i = 0; i < this.symbols.length; i += this.batchSize) {
            batches.push(this.symbols.slice(i, i + this.batchSize));
        }
        return batches;
    }

    async processBatch(batch, batchNumber, totalBatches) {
        const batchPromises = batch.map(symbol => this.scanSymbol(symbol));
        const batchResults = await Promise.allSettled(batchPromises);
        
        const successfulResults = batchResults
            .filter(result => result.status === 'fulfilled' && result.value.success)
            .map(result => result.value);

        const failedResults = batchResults
            .filter(result => result.status === 'fulfilled' && !result.value.success)
            .map(result => result.value);

        this.results.push(...successfulResults, ...failedResults);
        this.completed += batch.length;

        this.updateProgress(batch, batchNumber, totalBatches);
    }

    async scanSymbol(symbol) {
        try {
            // استفاده از روت‌های جدید
            const endpoint = this.mode === 'ai' ? 
                `/api/raw/${symbol}` : `/api/processed/${symbol}`;
            
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 15000);
            
            const response = await fetch(endpoint, {
                signal: controller.signal,
                headers: {
                    'Cache-Control': 'no-cache'
                }
            });
            
            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();
            
            return {
                symbol,
                success: true,
                data: data.data || data, // سازگاری با ساختار جدید
                timestamp: new Date().toISOString(),
                scanMode: this.mode
            };

        } catch (error) {
            return {
                symbol,
                success: false,
                error: error.message,
                timestamp: new Date().toISOString(),
                scanMode: this.mode
            };
        }
    }

    updateProgress(currentBatch, batchNumber, totalBatches) {
        const total = this.symbols.length;
        const percent = Math.round((this.completed / total) * 100);
        const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
        const speed = elapsed > 0 ? Math.round((this.completed / elapsed) * 60) : 0;

        this.onProgress?.({
            completed: this.completed,
            total,
            percent,
            elapsed,
            speed,
            currentBatch,
            batchNumber,
            totalBatches,
            mode: this.mode
        });
    }

    async batchScan(symbols, mode = 'basic') {
        // اسکن دسته‌ای مستقیم - برای استفاده سریع
        const endpoint = mode === 'ai' ? '/api/raw/batch' : '/api/processed/batch';
        
        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    symbols: symbols,
                    data_type: mode === 'ai' ? 'raw' : 'processed'
                })
            });

            if (!response.ok) {
                throw new Error(`Batch scan failed: ${response.status}`);
            }

            const result = await response.json();
            
            // تبدیل به فرمت سازگار
            const formattedResults = result.results.map(item => ({
                symbol: item.symbol,
                success: item.status === 'success',
                data: item.data,
                error: item.error,
                timestamp: new Date().toISOString(),
                scanMode: mode
            }));

            return formattedResults;

        } catch (error) {
            console.error('خطا در اسکن دسته‌ای:', error);
            throw error;
        }
    }

    cancel() {
        this.isCancelled = true;
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // ابزارهای کمکی
    getStats() {
        const successful = this.results.filter(r => r.success).length;
        const failed = this.results.filter(r => !r.success).length;
        const totalTime = this.startTime ? Date.now() - this.startTime : 0;

        return {
            total: this.results.length,
            successful,
            failed,
            successRate: this.results.length > 0 ? (successful / this.results.length * 100).toFixed(1) + '%' : '0%',
            totalTime: Math.round(totalTime / 1000) + 's',
            mode: this.mode,
            batchSize: this.batchSize
        };
    }

    clear() {
        this.results = [];
        this.completed = 0;
        this.isCancelled = false;
    }
}
