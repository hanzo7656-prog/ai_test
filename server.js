const express = require('express');
const axios = require('axios');
const app = express();
const PORT = process.env.PORT || 3000;

// Middleware ساده
app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    next();
});

// Route اصلی
app.get('/api/coins', async (req, res) => {
    try {
        console.log('📡 درخواست دریافت شد');
        const response = await axios.get('https://api.coinstats.app/public/v1/coins?limit=100');
        res.json(response.data);
    } catch (error) {
        console.error('❌ خطا:', error);
        res.status(500).json({ error: 'خطا در دریافت داده' });
    }
});

// Route سلامت
app.get('/health', (req, res) => {
    res.json({ status: 'OK', time: new Date().toISOString() });
});

app.listen(PORT, () => {
    console.log(🚀 سرور اجرا شد روی پورت ${PORT});
});
