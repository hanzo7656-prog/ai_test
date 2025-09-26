const express = require('express');
const axios = require('axios');
const app = express();
const PORT = process.env.PORT || 3000;

// Middleware ساده برای CORS
app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Headers', '*');
    next();
});

// ✅ Route اصلی - صفحه اول
app.get('/', (req, res) => {
    res.json({ 
        message: 'Server is working!',
        endpoints: {
            health: '/health',
            coins: '/api/coins'
        }
    });
});

// ✅ Route سلامت - این را اضافه کنید
app.get('/health', (req, res) => {
    res.json({ 
        status: 'OK', 
        timestamp: new Date().toISOString(),
        server: 'Crypto Scanner API'
    });
});

// ✅ Route دریافت داده ارزها
app.get('/api/coins', async (req, res) => {
    try {
        const limit = req.query.limit || 100;
        console.log('🌐 Fetching data from CoinStats...');
        
        const response = await axios.get(`https://api.coinstats.app/public/v1/coins?limit=${limit}`);
        
        res.json({
            success: true,
            data: response.data,
            source: 'CoinStats API',
            timestamp: new Date().toISOString()
        });
        
    } catch (error) {
        console.error('❌ Error:', error.message);
        res.status(500).json({
            success: false,
            error: error.message,
            timestamp: new Date().toISOString()
        });
    }
});

// راه‌اندازی سرور
app.listen(PORT, () => {
    console.log(`🚀 Server running on port ${PORT}`);
    console.log(`✅ Health check: http://localhost:${PORT}/health`);
    console.log(`✅ API endpoint: http://localhost:${PORT}/api/coins`);
});
