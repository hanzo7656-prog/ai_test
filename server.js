const express = require('express');
const axios = require('axios');
const app = express();
const PORT = process.env.PORT || 3000;

// ✅ حتماً این خط را اضافه کنید
app.use(express.json());

// ✅ CORS middleware
app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Headers', '*');
    res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE');
    next();
});

// ✅ Route اصلی
app.get('/', (req, res) => {
    res.json({ 
        message: 'Server is working!',
        endpoints: {
            health: '/health',
            coins: '/api/coins'
        }
    });
});

// ✅ Route سلامت - این حتماً باید باشد
app.get('/health', (req, res) => {
    res.json({ 
        status: 'OK', 
        message: 'Server is healthy',
        timestamp: new Date().toISOString()
    });
});

// ✅ Route دریافت داده ارزها
app.get('/api/coins', async (req, res) => {
    try {
        const limit = req.query.limit || 100;
        console.log('Fetching crypto data...');
        
        const response = await axios.get(`https://api.coinstats.app/public/v1/coins?limit=${limit}`);
        
        res.json({
            success: true,
            data: response.data,
            count: response.data.coins.length,
            timestamp: new Date().toISOString()
        });
        
    } catch (error) {
        console.error('Error:', error.message);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// ✅ Route جدید برای تست ساده
app.get('/test', (req, res) => {
    res.json({ message: 'Test endpoint works!' });
});

app.listen(PORT, () => {
    console.log(`🚀 Server running on port ${PORT}`);
});
