const WebSocket = require('ws');
const constants = require('../config/constants');

class WebSocketManager {
    constructor(gistManager) {
        this.ws = null;
        this.connected = false;
        this.realtimeData = {};
        this.subscribedPairs = new Set();
        this.gistManager = gistManager;  // ✅ دریافت gistManager
        this.connect();
    }

    connect() {
        try {
            this.ws = new WebSocket('wss://www.lbkex.net/ws/V2/');
            
            this.ws.on('open', () => {
                console.log('✔ WebSocket connected to LBank');
                this.connected = true;
                this.subscribeToAllPairs();
            });

            this.ws.on('message', (data) => {
                try {
                    const message = JSON.parse(data.toString());
                    if (message.type === 'tick' && message.tick) {
                        const symbol = message.pair;
                        const tickData = message.tick;
                        const currentPrice = parseFloat(tickData.latest);

                        // ✅ ذخیره در Gist (آنکامنت شده)
                        if (this.gistManager) {
                            this.gistManager.addPrice(symbol, currentPrice);
                        }

                        // ذخیره در داده real-time
                        this.realtimeData[symbol] = {
                            symbol: symbol,
                            price: currentPrice,
                            high_24h: parseFloat(tickData.high),
                            low_24h: parseFloat(tickData.low),
                            volume: parseFloat(tickData.vol),
                            change: parseFloat(tickData.change),
                            timestamp: message.TS,
                            last_updated: new Date().toISOString()
                        };

                        console.log(`📊 ${symbol}: $${currentPrice}`);
                    }
                } catch (error) {
                    console.error('✗ WebSocket message processing error', error);
                }
            });

            this.ws.on('error', (error) => {
                console.error('✗ WebSocket error', error);
                this.connected = false;
            });

            this.ws.on('close', (code, reason) => {
                console.log(`△ WebSocket disconnected - Code: ${code}, Reason: ${reason}`);
                this.connected = false;
                setTimeout(() => {
                    console.log('🔄 Attempting WebSocket reconnection...');
                    this.connect();
                }, 5000);
            });
        } catch (error) {
            console.error('WebSocket connection error', error);
            setTimeout(() => this.connect(), 10000);
        }
    }

    subscribeToAllPairs() {
        if (this.connected && this.ws) {
            console.log(`📡 Subscribing to ${constants.ALL_TRADING_PAIRS.length} trading pairs`);
            const batchSize = 10;
            
            for (let i = 0; i < constants.ALL_TRADING_PAIRS.length; i += batchSize) {
                setTimeout(() => {
                    const batch = constants.ALL_TRADING_PAIRS.slice(i, i + batchSize);
                    this.subscribeBatch(batch);
                }, i * 100);
            }
        }
    }

    subscribeBatch(pairs) {
        if (!this.ws) return;
        
        pairs.forEach(pair => {
            const subscription = {
                "action": "subscribe",
                "subscribe": "tick",
                "pair": pair
            };
            this.ws.send(JSON.stringify(subscription));
            this.subscribedPairs.add(pair);
        });
        console.log(`✅ Subscribed to ${pairs.length} pairs`);
    }

    getRealtimeData() {
        return this.realtimeData;
    }

    getConnectionStatus() {
        return {
            connected: this.connected,
            active_coins: Object.keys(this.realtimeData).length,
            total_subscribed: this.subscribedPairs.size,
            coins: Object.keys(this.realtimeData)
        };
    }

    // اضافه کردن متد برای تست
    testGistConnection() {
        if (this.gistManager) {
            const status = this.gistManager.getStatus();
            console.log('🧪 Gist Manager Test:', status);
            return status;
        }
        return { error: 'Gist Manager not available' };
    }
}

module.exports = WebSocketManager;
