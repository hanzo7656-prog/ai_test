// سیستم اسکن پیشرفته VortexAI - سازگار با روت‌های جدید
class ScanSession {
    constructor(options) {
        this.symbols = options.symbols;
        this.mode = options.mode; // 'ai' یا 'basic'
        this.batchSize = options.batchSize || 25;
        this.onProgress = options.onProgress;
        this.onComplete = options.onComplete;
        this.onError = options.onError;
        
        this.isCancelled = false;
        this.startTime = null;
        this.completed = 0;
        this.results = [];
        this.currentBatch = [];
        this.failedScans = 0;
        
        // آمار عملکرد
        this.performanceStats = {
            totalRequests: 0,
            successfulRequests: 0,
            failedRequests: 0,
            averageResponseTime: 0,
            totalTime: 0
        };
        
        console.log(`✅ ScanSession created: ${this.symbols.length} symbols, mode: ${this.mode}`);
    }

    async start() {
        this.startTime = Date.now();
        this.isCancelled = false;
        this.completed = 0;
        this.results = [];
        this.failedScans = 0;
        this.performanceStats = {
            totalRequests: 0,
            successfulRequests: 0,
            failedRequests: 0,
            averageResponseTime: 0,
            totalTime: 0
        };

        try {
            const batches = this.createBatches();
            console.log(`🚀 Starting scan with ${batches.length} batches`);
            
            for (let i = 0; i < batches.length; i++) {
                if (this.isCancelled) {
                    console.log('⏹️ Scan cancelled by user');
                    break;
                }

                const batch = batches[i];
                this.currentBatch = batch;
                
                await this.processBatch(batch, i + 1, batches.length);
                
                // تاخیر بین batchها برای کاهش فشار
                if (i < batches.length - 1 && !this.isCancelled) {
                    await this.delay(1000);
                }
            }

            if (!this.isCancelled) {
                this.performanceStats.totalTime = Date.now() - this.startTime;
                this.calculatePerformanceStats();
                
                console.log(`✅ Scan completed: ${this.results.length} results, ${this.failedScans} failed`);
                this.onComplete?.(this.results);
            } else {
                console.log('⏹️ Scan was cancelled');
            }

        } catch (error) {
            console.error('❌ Scan error:', error);
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
        const batchStartTime = Date.now();
        console.log(`🔄 Processing batch ${batchNumber}/${totalBatches} with ${batch.length} symbols`);
        
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
        this.failedScans += failedResults.length;

        // آپدیت آمار
        this.performanceStats.successfulRequests += successfulResults.length;
        this.performanceStats.failedRequests += failedResults.length;
        this.performanceStats.totalRequests += batch.length;

        const batchTime = Date.now() - batchStartTime;
        this.updateProgress(batch, batchNumber, totalBatches, batchTime);

        console.log(`✅ Batch ${batchNumber} completed: ${successfulResults.length} success, ${failedResults.length} failed`);
    }

    async scanSymbol(symbol) {
        const startTime = Date.now();
        
        try {
            // استفاده از روت‌های جدید
            const endpoint = this.mode === 'ai' ? 
                `/api/raw/${symbol}` : `/api/processed/${symbol}`;
            
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 15000);
            
            console.log(`🔍 Scanning ${symbol} via ${endpoint}`);
            
            const response = await fetch(endpoint, {
                signal: controller.signal,
                headers: {
                    'Cache-Control': 'no-cache',
                    'Accept': 'application/json'
                }
            });
            
            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            const responseTime = Date.now() - startTime;
            
            // اعتبارسنجی پاسخ
            if (!this.validateResponse(data)) {
                throw new Error('Invalid response format');
            }
            
            console.log(`✅ ${symbol} scanned successfully in ${responseTime}ms`);
            
            return {
                symbol,
                success: true,
                data: data.data || data,
                timestamp: new Date().toISOString(),
                scanMode: this.mode,
                responseTime: responseTime,
                source: endpoint
            };

        } catch (error) {
            const responseTime = Date.now() - startTime;
            console.error(`❌ Failed to scan ${symbol}:`, error.message);
            
            return {
                symbol,
                success: false,
                error: error.message,
                timestamp: new Date().toISOString(),
                scanMode: this.mode,
                responseTime: responseTime,
                source: 'error'
            };
        }
    }

    validateResponse(data) {
        // اعتبارسنجی ساختار پاسخ
        if (!data) return false;
        
        if (this.mode === 'ai') {
            // اعتبارسنجی داده‌های AI
            return data.data && data.data.market_data;
        } else {
            // اعتبارسنجی داده‌های پردازش شده
            return data.data && data.data.display_data;
        }
    }

    updateProgress(currentBatch, batchNumber, totalBatches, batchTime) {
        const total = this.symbols.length;
        const percent = Math.round((this.completed / total) * 100);
        const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
        const speed = elapsed > 0 ? Math.round((this.completed / elapsed) * 60) : 0;
        
        // محاسبه زمان باقی‌مانده
        const remainingTime = speed > 0 ? Math.round((total - this.completed) / speed * 60) : 0;

        const progressData = {
            completed: this.completed,
            total,
            percent,
            elapsed,
            remaining: remainingTime,
            speed,
            currentBatch,
            batchNumber,
            totalBatches,
            batchTime,
            failed: this.failedScans,
            mode: this.mode,
            performance: this.getCurrentPerformance()
        };

        this.onProgress?.(progressData);
    }

    getCurrentPerformance() {
        const successRate = this.performanceStats.totalRequests > 0 ? 
            (this.performanceStats.successfulRequests / this.performanceStats.totalRequests) * 100 : 0;
            
        return {
            successRate: Math.round(successRate),
            totalRequests: this.performanceStats.totalRequests,
            successful: this.performanceStats.successfulRequests,
            failed: this.performanceStats.failedRequests,
            averageResponseTime: this.performanceStats.averageResponseTime
        };
    }

    calculatePerformanceStats() {
        if (this.results.length > 0) {
            const totalResponseTime = this.results.reduce((sum, result) => sum + (result.responseTime || 0), 0);
            this.performanceStats.averageResponseTime = Math.round(totalResponseTime / this.results.length);
        }
    }

    // اسکن دسته‌ای مستقیم - برای استفاده سریع
    async batchScan(symbols, mode = 'basic') {
        const endpoint = mode === 'ai' ? '/api/raw/batch' : '/api/processed/batch';
        const startTime = Date.now();
        
        try {
            console.log(`🚀 Starting batch scan for ${symbols.length} symbols (${mode})`);
            
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({
                    symbols: symbols,
                    data_type: mode === 'ai' ? 'raw' : 'processed'
                })
            });

            if (!response.ok) {
                throw new Error(`Batch scan failed: ${response.status} ${response.statusText}`);
            }

            const result = await response.json();
            const totalTime = Date.now() - startTime;
            
            console.log(`✅ Batch scan completed in ${totalTime}ms: ${result.successful || 0} successful`);
            
            // تبدیل به فرمت سازگار
            const formattedResults = result.results.map(item => ({
                symbol: item.symbol,
                success: item.status === 'success',
                data: item.data,
                error: item.error,
                timestamp: new Date().toISOString(),
                scanMode: mode,
                responseTime: totalTime,
                source: 'batch'
            }));

            return {
                results: formattedResults,
                stats: {
                    total: result.total_symbols || symbols.length,
                    successful: result.successful || 0,
                    failed: result.failed || 0,
                    totalTime: totalTime
                },
                rawResponse: result
            };

        } catch (error) {
            console.error('❌ Batch scan error:', error);
            throw error;
        }
    }

    // اسکن تکی سریع
    async quickScan(symbol, mode = 'basic') {
        return await this.scanSymbol(symbol, mode);
    }

    cancel() {
        this.isCancelled = true;
        console.log('⏹️ Scan cancellation requested');
    }

    pause() {
        this.isCancelled = true;
        console.log('⏸️ Scan paused');
    }

    resume() {
        if (this.isCancelled) {
            this.isCancelled = false;
            console.log('▶️ Scan resumed');
            // می‌توانید اینجا منطق restart اضافه کنید
        }
    }

    getStats() {
        const successful = this.results.filter(r => r.success).length;
        const failed = this.results.filter(r => !r.success).length;
        const totalTime = this.startTime ? Date.now() - this.startTime : 0;
        const successRate = this.results.length > 0 ? (successful / this.results.length * 100) : 0;

        return {
            total: this.results.length,
            successful,
            failed,
            successRate: Math.round(successRate) + '%',
            totalTime: Math.round(totalTime / 1000) + 's',
            mode: this.mode,
            batchSize: this.batchSize,
            performance: this.performanceStats
        };
    }

    getResultsBySymbol(symbol) {
        return this.results.filter(result => result.symbol === symbol);
    }

    getSuccessfulResults() {
        return this.results.filter(result => result.success);
    }

    getFailedResults() {
        return this.results.filter(result => !result.success);
    }

    clear() {
        this.results = [];
        this.completed = 0;
        this.isCancelled = false;
        this.performanceStats = {
            totalRequests: 0,
            successfulRequests: 0,
            failedRequests: 0,
            averageResponseTime: 0,
            totalTime: 0
        };
        console.log('🧹 Scan session cleared');
    }

    exportResults(format = 'json') {
        if (this.results.length === 0) {
            throw new Error('No results to export');
        }

        const exportData = {
            metadata: {
                exportDate: new Date().toISOString(),
                totalResults: this.results.length,
                successfulResults: this.getSuccessfulResults().length,
                failedResults: this.getFailedResults().length,
                scanMode: this.mode,
                batchSize: this.batchSize
            },
            results: this.results
        };

        if (format === 'json') {
            return JSON.stringify(exportData, null, 2);
        } else if (format === 'csv') {
            return this.convertToCSV(exportData.results);
        } else {
            throw new Error('Unsupported export format');
        }
    }

    convertToCSV(results) {
        const headers = ['Symbol', 'Success', 'Scan Mode', 'Response Time', 'Timestamp', 'Error'];
        const rows = results.map(result => [
            result.symbol,
            result.success ? 'Yes' : 'No',
            result.scanMode,
            result.responseTime + 'ms',
            result.timestamp,
            result.error || 'N/A'
        ]);

        return [headers, ...rows].map(row => row.join(',')).join('\n');
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// ایجاد نمونه جهانی برای دسترسی آسان
if (typeof window !== 'undefined') {
    window.ScanSession = ScanSession;
}
