const express = require('express');
const axios = require('axios');
const app = express();
const PORT = process.env.PORT || 3000;

// Middleware ساده
app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    next();
});

// Route تست
app.get('/', (req, res) => {
    res.json({ message: '✅ Server is working!', time: new Date().toISOString() });
});

// Route سلامت
app.get('/health', (req, res) => {
    res.json({ status: 'OK', timestamp: new Date().toISOString() });
});

// Route اصلی (ساده‌شده)
app.get('/api/coins', async (req, res) => {
    try {
        console.log('🌐 Fetching data...');
        const response = await axios.get('https://api.coinstats.app/public/v1/coins?limit=10');
        res.json({ 
            success: true, 
            data: response.data,
            source: 'CoinStats API'
        });
    } catch (error) {
        console.error('Error:', error.message);
        res.status(500).json({ 
            success: false, 
            error: error.message 
        });
    }
});

app.listen(PORT, () => {
    console.log(`🚀 Server running on port ${PORT}`);
});
