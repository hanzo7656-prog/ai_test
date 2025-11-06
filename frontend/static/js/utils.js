// توابع کمکی عمومی

function formatPrice(price) {
    if (price === 0) return '0.00';
    if (price < 0.01) return price.toFixed(6);
    if (price < 1) return price.toFixed(4);
    if (price < 1000) return price.toFixed(2);
    return price.toLocaleString('en-US', { maximumFractionDigits: 2 });
}

function formatNumber(num) {
    if (num === 0) return '0';
    if (num < 1000) return num.toString();
    if (num < 1000000) return (num / 1000).toFixed(1) + 'K';
    if (num < 1000000000) return (num / 1000000).toFixed(1) + 'M';
    if (num < 1000000000000) return (num / 1000000000).toFixed(1) + 'B';
    return (num / 1000000000000).toFixed(1) + 'T';
}

function getDataFreshness(timestamp) {
    const now = new Date();
    const dataTime = new Date(timestamp);
    const diffMinutes = Math.round((now - dataTime) / (1000 * 60));
    
    if (diffMinutes < 1) return 'همین لحظه';
    if (diffMinutes < 5) return 'دقایقی پیش';
    if (diffMinutes < 30) return 'اخیراً';
    return 'قدیمی';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function stringToHash(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        const char = str.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash;
    }
    return Math.abs(hash);
}

function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}

function formatUptime(seconds) {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    return `${days}d ${hours}h`;
}

function downloadFile(filename, content) {
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function getCoinSymbol(symbol) {
    const symbolsMap = {
        'bitcoin': '₿',
        'ethereum': 'Ξ',
        'tether': '₮',
        'ripple': 'X',
        'binancecoin': 'BNB',
        'solana': 'SOL',
        'usd-coin': 'USDC',
        'staked-ether': 'ETH2',
        'tron': 'TRX',
        'dogecoin': 'DOGE',
        'cardano': 'ADA',
        'polkadot': 'DOT',
        'chainlink': 'LINK',
        'litecoin': 'LTC',
        'bitcoin-cash': 'BCH'
    };
    return symbolsMap[symbol] || symbol.substring(0, 3).toUpperCase();
}

export {
    formatPrice,
    formatNumber,
    getDataFreshness,
    escapeHtml,
    stringToHash,
    formatTime,
    formatUptime,
    downloadFile,
    getCoinSymbol
};
