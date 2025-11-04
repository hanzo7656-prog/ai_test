// سیستم اسکن بهینه‌شده VortexAI
class OptimizedScanner {
    constructor() {
        this.isScanning = false;
        this.currentScanId = null;
        this.batchSize = 25;
        this.scanStartTime = null;
        
        // لیست 100 ارز برتر
        this.top100Symbols = [
            "bitcoin", "ethereum", "tether", "ripple", "binance-coin",
            "solana", "usd-coin", "staked-ether", "tron", "dogecoin",
            "cardano", "figure-heloc", "wrapped-bitcoin", "chainlink", 
            "hyperliquid", "bitcoin-cash", "wrapped-eeth", "ethena-usde",
            "stellar", "whitebit", "sui", "hedera-hashgraph", "avalanche-2",
            "litecoin", "zcash", "monero", "shiba-inu", "the-open-network",
            "dai", "crypto-com-chain", "polkadot", "bittensor", "memecore",
            "mantle", "uniswap", "world-liberty-financial", "aave",
            "blackrock-usd-institutional-digital-liquidity-fund", "internet-computer",
            "paypal-usd", "bitget-token", "okb", "near", "pepe", "ethena",
            "ethereum-classic", "falcon-finance", "tether-gold", "aptos",
            "ondo-finance", "aster-2", "pi-network", "usdtb", "polygon-ecosystem-token",
            "worldcoin-wld", "kucoin-shares", "dash", "rocket-pool-eth",
            "binance-staked-sol", "arbitrum", "official-trump", "gatechain-token",
            "algorand", "pump-fun", "syrupusdt", "pax-gold", "stakewise-v3-oseth",
            "syrupusdc", "function-fbtc", "liquid-staked-ethereum", "vechain",
            "cosmos", "story-2", "kaspa", "sky", "jupiter-exchange-solana",
            "flare-networks", "quant", "nexo", "filecoin", "ripple-usd",
            "render-token", "sei-network", "global-dollar", "first-digital-usd",
            "xinfin-network", "pudgy-penguins", "bonk", "virtual-protocol",
            "mantle-staked-ether", "morpho", "immutable-x", "hashnote-usyc",
            "fasttoken", "ousg", "pancakeswap-token", "aerodrome-finance",
            "cgeth-hashkey-cloud", "optimism", "ondo-us-dollar-yield"
        ];
    }

    // اسکن هوشمند - تشخیص خودکار تکی/دسته‌ای
    async smartScan(selectedSymbols = [], isAIMode = false) {
        if (this.isScanning) {
            alert('اسکن در حال انجام است!');
            return;
        }

        this.isScanning = true;
        this.scanStartTime = Date.now();
        this.currentScanId = 'scan_' + Date.now();
        
        const symbolsToScan = selectedSymbols.length > 0 ? selectedSymbols : this.top100Symbols;
        const scanType = selectedSymbols.length === 1 ? 'تکی' : 
                        selectedSymbols.length > 1 ? 'دسته‌ای انتخابی' : 'دسته‌ای کامل';

        // شروع لودینگ هوشمند
        smartLoading.start({
            total: symbolsToScan.length,
            scanType: scanType,
            isAIMode: isAIMode,
            symbols: symbolsToScan
        });

        try {
            if (symbolsToScan.length === 1) {
                // اسکن تکی
                await this.singleScan(symbolsToScan[0], isAIMode);
            } else {
                // اسکن دسته‌ای
                await this.batchScan(symbolsToScan, isAIMode);
            }
        } catch (error) {
            console.error('خطا در اسکن:', error);
            smartLoading.showError('خطا در اسکن: ' + error.message);
        } finally {
            this.isScanning = false;
            smartLoading.complete();
        }
    }

    // اسکن تکی بهینه‌شده
    async singleScan(symbol, isAIMode) {
        const cacheKey = `scan_${isAIMode ? 'ai' : 'manual'}_${symbol}`;
        const cached = cacheManager.get(cacheKey);
        
        if (cached && !isAIMode) { // فقط برای Manual از کش استفاده کن
            smartLoading.updateProgress(1, 1, [symbol], [symbol]);
            this.displayResults([cached]);
            return;
        }

        const endpoint = isAIMode ? `/api/scan/ai/${symbol}` : `/api/scan/basic/${symbol}`;
        
        try {
            smartLoading.updateCurrentScanning([symbol]);
            
            const response = await fetch(endpoint);
            if (!response.ok) throw new Error('خطا در دریافت داده');
            
            const data = await response.json();
            
            // کش کردن نتیجه
            if (!isAIMode) {
                cacheManager.set(cacheKey, data, 5 * 60 * 1000); // 5 دقیقه
            }
            
            smartLoading.updateProgress(1, 1, [symbol], [symbol]);
            this.displayResults([data]);
            
        } catch (error) {
            throw new Error(`خطا در اسکن ${symbol}: ${error.message}`);
        }
    }

    // اسکن دسته‌ای بهینه‌شده
    async batchScan(symbols, isAIMode) {
        const total = symbols.length;
        let completed = 0;
        const completedSymbols = [];
        
        // تقسیم به دسته‌های 25 تایی
        const batches = [];
        for (let i = 0; i < symbols.length; i += this.batchSize) {
            batches.push(symbols.slice(i, i + this.batchSize));
        }

        const results = [];

        for (let batchIndex = 0; batchIndex < batches.length; batchIndex++) {
            const batchSymbols = batches[batchIndex];
            const batchResults = [];
            
            smartLoading.updateBatchInfo(batchIndex + 1, batches.length);

            // اسکن موازی در هر دسته
            const promises = batchSymbols.map(async (symbol) => {
                try {
                    smartLoading.updateCurrentScanning(batchSymbols);
                    
                    const endpoint = isAIMode ? `/api/scan/ai/${symbol}` : `/api/scan/basic/${symbol}`;
                    const response = await fetch(endpoint);
                    
                    if (response.ok) {
                        const data = await response.json();
                        batchResults.push(data);
                        completedSymbols.push(symbol);
                        
                        // کش کردن نتیجه برای Manual
                        if (!isAIMode) {
                            const cacheKey = `scan_manual_${symbol}`;
                            cacheManager.set(cacheKey, data, 5 * 60 * 1000);
                        }
                    }
                } catch (error) {
                    console.error(`خطا در اسکن ${symbol}:`, error);
                    batchResults.push(this.createErrorResult(symbol, error.message));
                } finally {
                    completed++;
                    smartLoading.updateProgress(completed, total, batchSymbols, completedSymbols);
                }
            });

            await Promise.all(promises);
            results.push(...batchResults);
            
            // نمایش نتایج هر دسته
            this.displayResults(results);
        }

        return results;
    }

    createErrorResult(symbol, error) {
        return {
            status: "error",
            symbol: symbol,
            error: error,
            timestamp: new Date().toISOString()
        };
    }

    displayResults(results) {
        const container = document.getElementById('scanResults');
        if (!container) return;

        const cards = results.map(result => this.createSymbolCard(result)).join('');
        container.innerHTML = cards || '<div class="no-results">هیچ نتیجه‌ای یافت نشد</div>';
        
        // آپدیت شمارنده نتایج
        const countElement = document.getElementById('resultsCount');
        if (countElement) {
            countElement.textContent = `${results.length} ارز`;
        }
    }

    createSymbolCard(data) {
        if (data.status === "error") {
            return this.createErrorCard(data.symbol, data.error);
        }

        const symbol = data.symbol;
        const isAI = data.data_type === "raw";
        const displayData = data.data?.display_data || {};
        const analysis = data.data?.analysis || {};
        
        const price = displayData.price || 0;
        const change = displayData.price_change_24h || displayData.priceChange1d || 0;
        const changeClass = change > 0 ? 'positive' : 'negative';
        const changeSymbol = change > 0 ? '▲' : '▼';

        const signal = analysis.signal || 'HOLD';
        const confidence = analysis.confidence || 0.5;

        return `
            <div class="symbol-card" data-symbol="${symbol}" data-timestamp="${data.timestamp}">
                <div class="symbol-header">
                    <div class="coin-icon">${this.getCoinIcon(symbol)}</div>
                    <div class="symbol-info">
                        <div class="symbol-name">${symbol.toUpperCase()}</div>
                        <div class="symbol-fullname">${displayData.name || 'Unknown'}</div>
                    </div>
                    ${isAI ? '<span class="ai-badge">AI</span>' : ''}
                </div>

                <div class="price-section">
                    <div class="price-item">
                        <div class="price-label">قیمت</div>
                        <div class="price-value">$${price.toLocaleString()}</div>
                    </div>
                    <div class="price-item">
                        <div class="price-label">تغییر 24h</div>
                        <div class="price-value change ${changeClass}">
                            ${changeSymbol} ${Math.abs(change).toFixed(2)}%
                        </div>
                    </div>
                </div>

                <div class="analysis-section">
                    <div class="signal-badge ${this.getSignalClass(signal)}">
                        ${this.getSignalText(signal)}
                    </div>
                    
                    <div class="confidence-section">
                        <small>اعتماد: ${Math.round(confidence * 100)}%</small>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${confidence * 100}%"></div>
                        </div>
                    </div>

                    <div class="data-freshness">
                        <small>${this.getDataFreshness(data.timestamp)}</small>
                    </div>
                </div>
            </div>
        `;
    }

    createErrorCard(symbol, error) {
        return `
            <div class="symbol-card error">
                <div class="symbol-header">
                    <div class="coin-icon">❌</div>
                    <div class="symbol-info">
                        <div class="symbol-name">${symbol.toUpperCase()}</div>
                        <div class="symbol-fullname">خطا در دریافت داده</div>
                    </div>
                </div>
                <div class="error-message">${error}</div>
            </div>
        `;
    }

    getCoinIcon(symbol) {
        const icons = {
            'bitcoin': '₿', 'ethereum': 'Ξ', 'tether': '₮', 'ripple': 'X',
            'solana': 'S', 'cardano': 'A', 'polkadot': 'D', 'chainlink': '●'
        };
        return icons[symbol] || symbol.charAt(0).toUpperCase();
    }

    getSignalText(signal) {
        const signals = {
            'STRONG_BUY': 'خرید قوی', 'BUY': 'خرید', 'HOLD': 'نگهداری',
            'SELL': 'فروش', 'STRONG_SELL': 'فروش قوی'
        };
        return signals[signal] || signal;
    }

    getSignalClass(signal) {
        const classes = {
            'STRONG_BUY': 'signal-buy', 'BUY': 'signal-buy',
            'HOLD': 'signal-hold',
            'SELL': 'signal-sell', 'STRONG_SELL': 'signal-sell'
        };
        return classes[signal] || 'signal-hold';
    }

    getDataFreshness(timestamp) {
        const now = new Date();
        const dataTime = new Date(timestamp);
        const diffMinutes = Math.round((now - dataTime) / (1000 * 60));
        
        if (diffMinutes < 2) return '🟢 لحظه‌ای';
        if (diffMinutes < 5) return '🟡 چند دقیقه پیش';
        if (diffMinutes < 10) return '🟠 کهنه';
        return '🔴 قدیمی';
    }

    cancelScan() {
        this.isScanning = false;
        smartLoading.complete();
    }
}

// نمونه جهانی
const optimizedScanner = new OptimizedScanner();
