// static/js/dashboard.js - فقط داده واقعی از API
class Dashboard {
    constructor() {
        this.coinData = {};
        this.initializeDashboard();
        this.setupEventListeners();
        this.startRealTimeUpdates();
    }

    async initializeDashboard() {
        await this.loadRealMarketData();
        this.updateActiveSignals();
        this.updateSystemStatus();
        this.setupChart();
    }

    setupEventListeners() {
        document.getElementById('alertsList')?.addEventListener('click', () => {
            window.location.href = '/health#alerts';
        });

        document.getElementById('signalsList')?.addEventListener('click', () => {
            window.location.href = '/analysis';
        });

        document.querySelectorAll('.quick-card').forEach(card => {
            card.addEventListener('click', () => {
                const page = card.dataset.page;
                if (page) window.location.href = page;
            });
        });
    }

    async loadRealMarketData() {
        try {
            console.log('🔄 دریافت داده‌های واقعی از API...');
            
            // دریافت داده از AI Analysis API
            const response = await fetch('/api/ai/analysis?symbols=BTC,ETH,SOL,ADA&period=1h');
            
            if (!response.ok) {
                throw new Error(`API Error: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('📊 پاسخ API:', data);

            if (data.status === 'success' && data.analysis_report) {
                this.processRealData(data.analysis_report);
            } else {
                throw new Error('داده معتبر از API دریافت نشد');
            }

        } catch (error) {
            console.error('❌ خطا در دریافت داده:', error);
            this.showDataError('اتصال به API برقرار نشد');
        }
    }

    processRealData(analysisReport) {
        if (!analysisReport.symbol_analysis) {
            this.showDataError('داده‌های تحلیل در دسترس نیست');
            return;
        }

        // پردازش داده‌های واقعی
        Object.entries(analysisReport.symbol_analysis).forEach(([symbol, data]) => {
            if (data && typeof data.current_price === 'number') {
                this.coinData[symbol] = {
                    price: data.current_price,
                    change: this.calculatePriceChange(data),
                    confidence: data.ai_signal?.signals?.signal_confidence || 0,
                    signal: data.ai_signal?.signals?.primary_signal || 'HOLD'
                };
            }
        });

        // آپدیت نمایش با داده واقعی
        this.updatePriceDisplays();
    }

    calculatePriceChange(data) {
        // محاسبه تغییر قیمت از داده‌های تاریخی
        if (data.historical_data?.result && data.historical_data.result.length > 1) {
            const prices = data.historical_data.result
                .map(item => item.price || item.close || item.last)
                .filter(price => price && !isNaN(price));
            
            if (prices.length > 1) {
                const current = prices[prices.length - 1];
                const previous = prices[prices.length - 2];
                return ((current - previous) / previous) * 100;
            }
        }
        return 0;
    }

    updatePriceDisplays() {
        // آپدیت BTC قیمت
        const btcData = this.coinData['BTC'];
        if (btcData) {
            this.updatePriceDisplay('BTC', btcData.price, btcData.change);
        }

        // آپدیت ETH قیمت
        const ethData = this.coinData['ETH'];
        if (ethData) {
            // می‌تونیم ETH رو هم نمایش بدیم یا از نمادهای دیگه استفاده کنیم
        }
    }

    updatePriceDisplay(symbol, price, change) {
        const priceElement = document.querySelector('.quick-chart .current-price');
        const changeElement = document.querySelector('.quick-chart .price-change');
        const titleElement = document.querySelector('.quick-chart .section-header h2');
        
        if (priceElement && changeElement) {
            priceElement.textContent = `$${price.toLocaleString('en-US', {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2
            })}`;
            
            changeElement.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`;
            changeElement.className = `price-change ${change >= 0 ? 'positive' : 'negative'}`;
            
            if (titleElement) {
                titleElement.textContent = `📊 ${symbol}/USDT`;
            }
        }
    }

    updateActiveSignals() {
        // استفاده از داده‌های واقعی برای سیگنال‌ها
        this.renderRealSignals();
    }

    renderRealSignals() {
        const container = document.getElementById('signalsList');
        if (!container) return;

        const signals = [];
        
        // استفاده از داده‌های واقعی از coinData
        Object.entries(this.coinData).forEach(([symbol, data]) => {
            if (data.price && Math.abs(data.change) > 0.1) { // فیلتر تغییرات معنادار
                signals.push({
                    symbol: symbol,
                    name: this.getCoinName(symbol),
                    price: data.price,
                    change: data.change,
                    type: data.change >= 0 ? 'bullish' : 'bearish',
                    confidence: Math.round((data.confidence || 0.5) * 100)
                });
            }
        });

        // اگر داده واقعی نداریم، هیچ چیزی نمایش ندهیم
        if (signals.length === 0) {
            container.innerHTML = '<div class="no-data">در حال دریافت داده‌های بازار...</div>';
            return;
        }

        // مرتب‌سازی بر اساس بیشترین تغییر
        signals.sort((a, b) => Math.abs(b.change) - Math.abs(a.change));

        container.innerHTML = signals.map(signal => `
            <div class="signal-item ${signal.type}" onclick="window.location.href='/analysis?symbol=${signal.symbol}'">
                <div class="signal-info">
                    <div class="signal-symbol">${signal.symbol}</div>
                    <div class="signal-name">${signal.name}</div>
                </div>
                <div class="signal-price">$${signal.price.toLocaleString()}</div>
                <div class="signal-change ${signal.change >= 0 ? 'positive' : 'negative'}">
                    ${signal.change >= 0 ? '+' : ''}${signal.change.toFixed(2)}%
                </div>
                <div class="signal-confidence">${signal.confidence}%</div>
            </div>
        `).join('');
    }

    getCoinName(symbol) {
        const names = {
            'BTC': 'Bitcoin',
            'ETH': 'Ethereum', 
            'SOL': 'Solana',
            'ADA': 'Cardano',
            'DOT': 'Polkadot',
            'LINK': 'Chainlink',
            'BNB': 'Binance Coin',
            'XRP': 'Ripple'
        };
        return names[symbol] || symbol;
    }

    updateSystemStatus() {
        // بررسی وضعیت واقعی سیستم از Health API
        this.checkSystemStatus();
    }

    async checkSystemStatus() {
        try {
            const response = await fetch('/api/system/health');
            if (!response.ok) throw new Error('Health API error');
            
            const data = await response.json();
            this.renderSystemStatus(data);
        } catch (error) {
            console.error('Error checking system status:', error);
            this.renderSystemStatus(null);
        }
    }

    renderSystemStatus(healthData) {
        const container = document.querySelector('.status-grid');
        if (!container) return;

        const statusItems = [
            { 
                label: 'API CoinStats', 
                value: healthData?.api_health === 'healthy' ? 'متصل' : 'قطع',
                status: healthData?.api_health === 'healthy' ? 'connected' : 'disconnected'
            },
            { 
                label: 'مدل AI', 
                value: healthData?.ai_status === 'active' ? 'فعال' : 'غیرفعال',
                status: healthData?.ai_status === 'active' ? 'active' : 'disconnected'
            },
            { 
                label: 'WebSocket', 
                value: healthData?.websocket_status === 'connected' ? 'متصل' : 'قطع',
                status: healthData?.websocket_status === 'connected' ? 'connected' : 'disconnected'
            },
            { 
                label: 'دقت پیش‌بینی', 
                value: healthData?.ai_accuracy ? `${Math.round(healthData.ai_accuracy)}%` : 'درحال محاسبه',
                status: 'normal'
            }
        ];

        container.innerHTML = statusItems.map(item => `
            <div class="status-item">
                <div class="status-label">${item.label}</div>
                <div class="status-value ${item.status}">${item.value}</div>
            </div>
        `).join('');
    }

    setupChart() {
        // نمودار با داده واقعی
        this.loadRealChartData();
    }

    async loadRealChartData() {
        try {
            const response = await fetch('/api/ai/analysis?symbols=BTC&period=24h');
            if (!response.ok) throw new Error('Chart API error');
            
            const data = await response.json();
            if (data.status === 'success') {
                this.renderRealChart(data.analysis_report);
            } else {
                this.showChartError('داده نمودار در دسترس نیست');
            }
        } catch (error) {
            console.error('Error loading chart data:', error);
            this.showChartError('خطا در دریافت داده نمودار');
        }
    }

    renderRealChart(analysisReport) {
        const container = document.getElementById('btcChart');
        if (!container) return;

        const btcData = analysisReport.symbol_analysis?.BTC;
        if (!btcData) {
            this.showChartError('داده BTC یافت نشد');
            return;
        }

        const prices = this.extractPricesFromData(btcData);
        if (prices.length === 0) {
            this.showChartError('داده قیمتی موجود نیست');
            return;
        }

        this.renderChart(container, prices);
    }

    extractPricesFromData(btcData) {
        try {
            // استخراج قیمت‌ها از داده‌های تاریخی
            if (btcData.historical_data?.result) {
                return btcData.historical_data.result
                    .slice(-20) // 20 داده آخر
                    .map(item => {
                        const price = item.price || item.close || item.last;
                        return price && !isNaN(price) ? parseFloat(price) : null;
                    })
                    .filter(price => price !== null);
            }
        } catch (error) {
            console.error('Error extracting prices:', error);
        }
        
        return [];
    }

    renderChart(container, prices) {
        if (prices.length === 0) {
            this.showChartError('داده‌ای برای نمایش موجود نیست');
            return;
        }

        const maxPrice = Math.max(...prices);
        const minPrice = Math.min(...prices);
        const range = maxPrice - minPrice || 1;

        container.innerHTML = '';
        const chart = document.createElement('div');
        chart.className = 'simple-chart';
        chart.style.cssText = `
            width: 100%;
            height: 100%;
            display: flex;
            align-items: flex-end;
            gap: 2px;
            padding: 10px;
        `;

        prices.forEach((price, index) => {
            const bar = document.createElement('div');
            const height = ((price - minPrice) / range) * 80;
            const isGreen = index === 0 || price >= prices[index - 1];
            
            bar.style.cssText = `
                flex: 1;
                height: ${height}%;
                background: ${isGreen ? 'var(--accent-success)' : 'var(--accent-danger)'};
                border-radius: 2px;
                opacity: ${0.6 + (index * 0.02)};
                transition: all 0.3s ease;
            `;
            
            bar.title = `$${price.toFixed(2)}`;
            chart.appendChild(bar);
        });

        container.appendChild(chart);
    }

    startRealTimeUpdates() {
        // بروزرسانی هر 15 ثانیه
        setInterval(() => {
            this.loadRealMarketData();
        }, 15000);

        // بروزرسانی وضعیت سیستم هر دقیقه
        setInterval(() => {
            this.checkSystemStatus();
        }, 60000);
    }

    showDataError(message) {
        const priceElement = document.querySelector('.quick-chart .current-price');
        const changeElement = document.querySelector('.quick-chart .price-change');
        
        if (priceElement) priceElement.textContent = '---';
        if (changeElement) {
            changeElement.textContent = message;
            changeElement.className = 'price-change error';
        }

        const signalsContainer = document.getElementById('signalsList');
        if (signalsContainer) {
            signalsContainer.innerHTML = `<div class="no-data">${message}</div>`;
        }
    }

    showChartError(message) {
        const container = document.getElementById('btcChart');
        if (container) {
            container.innerHTML = `<div class="chart-error">${message}</div>`;
        }
    }
}

// راه‌اندازی
document.addEventListener('DOMContentLoaded', () => {
    new Dashboard();
});
