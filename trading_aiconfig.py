from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class TradingConfig:
    """تنظیمات اصلی سیستم تحلیل تکنیکال"""
    
    # نمادهای تحت پوشش - استفاده از لیست 100 تایی از app.js
    SYMBOLS: List[str] = field(default_factory=list)  # ✅ اصلاح شد
    
    # تنظیمات معماری اسپارس
    SPARSE_NEURONS: int = 2500
    SPARSE_CONNECTIONS: int = 50
    TEMPORAL_SEQUENCE: int = 60
    INPUT_FEATURES: int = 5
    
    # تنظیمات آموزش
    TRAINING_EPOCHS: int = 30
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001
    
    # تنظیمات تحلیل تکنیکال
    TECHNICAL_INDICATORS: List[str] = field(default_factory=lambda: ['RSI', 'MACD', 'BBANDS', 'STOCH', 'ATR', 'OBV'])  # ✅ اصلاح شد
    LOOKBACK_DAYS: int = 100
    
    # تنظیمات اسکن
    SCAN_INTERVAL: int = 300
    CONFIDENCE_THRESHOLD: float = 0.7

@dataclass
class TechnicalConfig:
    """تنظیمات تحلیل تکنیکال"""
    RSI_PERIOD: int = 14
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    BB_PERIOD: int = 20
    BB_STD: int = 2
    ATR_PERIOD: int = 14

@dataclass  
class SparseConfig:
    """تنظیمات معماری اسپارس"""
    ACTIVITY_THRESHOLD: float = 0.01
    LEARNING_BOOST: float = 1.2
    NEUROGENESIS_RATE: float = 0.05

# لیست 100 ارز برتر از app.js
TOP_100_SYMBOLS = [
    "bitcoin", "ethereum", "tether", "ripple", "binancecoin",
    "solana", "usd-coin", "staked-ether", "tron", "dogecoin",
    "cardano", "polkadot", "chainlink", "litecoin", "bitcoin-cash",
    "stellar", "monero", "ethereum-classic", "vechain", "theta-token",
    "filecoin", "cosmos", "tezos", "aave", "eos",
    "okb", "crypto-com-chain", "algorand", "maker", "iota",
    "avalanche-2", "compound", "dash", "zcash", "neo",
    "kusama", "elrond-erd-2", "helium", "decentraland", "the-sandbox",
    "gala", "axie-infinity", "enjincoin", "render-token", "theta-fuel",
    "fantom", "klay-token", "waves", "arweave", "bittorrent",
    "huobi-token", "nexo", "celo", "qtum", "ravencoin",
    "basic-attention-token", "holotoken", "chiliz", "curve-dao-token", "kusama",
    "yearn-finance", "sushi", "uma", "balancer", "renbtc",
    "0x", "bancor", "loopring", "reserve-rights-token", "orchid",
    "nucypher", "livepeer", "api3", "uma", "badger-dao",
    "keep-network", "origin-protocol", "mirror-protocol", "radicle", "fetchtoken",
    "ocean-protocol", "dock", "request-network", "district0x", "gnosis",
    "kyber-network", "republic-protocol", "aeternity", "golem", "iostoken",
    "wax", "dent", "stormx", "funfair", "enigma",
    "singularitynet", "numeraire", "civic", "poa-network", "metal",
    "pillar", "bluzelle", "cybermiles", "datum", "edgeware"
]

# ایجاد نمونه‌های پیکربندی با لیست 100 تایی
trading_config = TradingConfig()
trading_config.SYMBOLS = TOP_100_SYMBOLS  # ✅ تنظیم لیست 100 تایی

technical_config = TechnicalConfig()
sparse_config = SparseConfig()
