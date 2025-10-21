# constants.py

# جفت‌ارزهای اصلی
MAIN_TRADING_PAIRS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT',
    'DOTUSDT', 'LTCUSDT', 'LINKUSDT', 'BCHUSDT', 'XLMUSDT'
]

# اندیکاتورها
TECHNICAL_INDICATORS = {
    'RSI': {'overbought': 70, 'oversold': 30},
    'MACD': {'signal_line': 9},
    'BOLLINGER': {'period': 20, 'std_dev': 2}
}

# سطوح احساسات
SENTIMENT_LEVELS = {
    'EXTREME_FEAR': (0, 25),
    'FEAR': (25, 45),
    'NEUTRAL': (45, 55),
    'GREED': (55, 75),
    'EXTREME_GREED': (75, 100)
}

# دسته‌بندی کوین‌ها
COIN_CATEGORIES = {
    'large_cap': ['BTC', 'ETH', 'BNB', 'ADA', 'XRP'],
    'mid_cap': ['DOT', 'LTC', 'LINK', 'BCH', 'XLM'],
    'defi': ['UNI', 'AAVE', 'COMP', 'MKR', 'SNX'],
    'nft': ['SAND', 'MANA', 'ENJ', 'FLOW', 'AXS']
}
