// سیستم اسکن پیشرفته VortexAI - نسخه بهینه شده
class ScanSession {
    constructor(options) {
        this.symbols = options.symbols || [];
        this.mode = options.mode || 'basic';
        this.batchSize = options.batchSize || 25;
        this.onProgress = options.onProgress || (() => {});
        this.onComplete = options.onComplete || (() => {});
        this.onError = options.onError || (() => {});
        
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
        console.log('🚀 Starting scan session...');
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
            // اعتبارسنجی سمبل‌ها
            const validSymbols = this.symbols.filter(symbol => 
                symbol && typeof symbol === 'string' && symbol.trim().length > 0
            );
            
            if (validSymbols.length === 0) {
                throw new Error('هیچ سمبل معتبری برای اسکن وجود ندارد');
            }

            const batches = this.createBatches(validSymbols);
            console.log(`🔄 Processing ${batches.length} batches with ${validSymbols.length} symbols`);
            
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
                    await VortexUtils.delay(500);
                }
            }

            if (!this.isCancelled) {
                this.performanceStats.totalTime = Date.now() - this.startTime;
                this.calculatePerformanceStats();
                
                console.log(`✅ Scan completed: ${this.results.length} results, ${this.failedScans} failed`);
                this.onComplete(this.results);
            } else {
                console.log('⏹️ Scan was cancelled');
            }

        } catch (error) {
            console.error('❌ Scan session error:', error);
            this.onError(error);
        }
    }

    createBatches(symbols) {
        const batches = [];
        for (let i = 0; i < symbols.length; i += this.batchSize) {
            batches.push(symbols.slice(i, i + this.batchSize));
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

        // پردازش نتایج rejected
        const rejectedResults = batchResults
            .filter(result => result.status === 'rejected')
            .map(result => ({
                symbol: 'unknown',
                success: false,
                error: result.reason?.message || 'Unknown error',
                timestamp: new Date().toISOString(),
                scanMode: this.mode,
                responseTime: 0
            }));

        this.results.push(...successfulResults, ...failedResults, ...rejectedResults);
        this.completed += batch.length;
        this.failedScans += (failedResults.length + rejectedResults.length);

        // آپدیت آمار
        this.performanceStats.successfulRequests += successfulResults.length;
        this.performanceStats.failedRequests += (failedResults.length + rejectedResults.length);
        this.performanceStats.totalRequests += batch.length;

        const batchTime = Date.now() - batchStartTime;
        this.updateProgress(batch, batchNumber, totalBatches, batchTime);

        console.log(`✅ Batch ${batchNumber} completed: ${successfulResults.length} success, ${failedResults.length + rejectedResults.length} failed`);
    }

    async scanSymbol(symbol) {
        const startTime = Date.now();
        
        try {
            // اعتبارسنجی سمبل
            if (!VortexUtils.isValidSymbol(symbol)) {
                throw new Error(`سمبل نامعتبر: ${symbol}`);
            }

            // انتخاب endpoint بر اساس حالت
            const endpoint = this.mode === 'ai' ? 
                `/api/raw/${symbol.toLowerCase()}` : 
                `/api/processed/${symbol.toLowerCase()}`;
            
            console.log(`🔍 Scanning ${symbol} via ${endpoint}`);
            
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 15000);
            
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
                throw new Error('پاسخ دریافتی معتبر نیست');
            }
            
            console.log(`✅ ${symbol} scanned successfully in ${responseTime}ms`);
            
            return {
                symbol: symbol.toLowerCase(),
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
                symbol: symbol.toLowerCase(),
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
        if (!data) return false;
        
        // بررسی ساختارهای مختلف پاسخ
        if (data.status === 'success' && data.data) {
            return true;
        }
        
        if (data.data_type && data.data) {
            return true;
        }
        
        // برای حالت fallback
        if (data.symbol || data.price !== undefined) {
            return true;
        }
        
        return false;
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

        this.onProgress(progressData);
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

    // اسکن دسته‌ای مستقیم
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
            const formattedResults = (result.results || []).map(item => ({
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
        return this.results.filter(result => result.symbol === symbol.toLowerCase());
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
                batchSize: this.batchSize,
                performance: this.performanceStats
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
        const headers = ['Symbol', 'Success', 'Price', 'Change%', 'Volume', 'MarketCap', 'Rank', 'Signal', 'Scan Mode', 'Response Time', 'Timestamp', 'Error'];
        const rows = results.map(result => {
            const data = result.data;
            let price = 0, change = 0, volume = 0, marketCap = 0, rank = null, signal = 'N/A';
            
            if (data && data.data) {
                const marketData = data.data.market_data || data.data.display_data || data.data;
                price = marketData.price || marketData.current_price || 0;
                change = marketData.priceChange1d || marketData.price_change_24h || 0;
                volume = marketData.volume || marketData.total_volume || 0;
                marketCap = marketData.marketCap || marketData.market_cap || 0;
                rank = marketData.rank || null;
                
                if (data.data.analysis) {
                    signal = data.data.analysis.signal || 'N/A';
                }
            }
            
            return [
                result.symbol.toUpperCase(),
                result.success ? 'Yes' : 'No',
                VortexUtils.formatPrice(price),
                change.toFixed(2) + '%',
                VortexUtils.formatNumber(volume),
                VortexUtils.formatNumber(marketCap),
                rank || 'N/A',
                signal,
                result.scanMode,
                result.responseTime + 'ms',
                result.timestamp,
                result.error || 'N/A'
            ];
        });

        return [headers, ...rows].map(row => 
            row.map(field => `"${field}"`).join(',')
        ).join('\n');
    }
}

// ایجاد نمونه جهانی
window.ScanSession = ScanSession;
