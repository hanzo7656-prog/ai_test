# 📁 config/phase5_config.py

PHASE5_CONFIG = {
    "system": {
        "name": "Crypto Market Analyzer Pro",
        "version": "1.0.0",
        "environment": "production",
        "log_level": "INFO",
        "analysis_interval": 300,  # 5 minutes
        "max_memory_usage_mb": 512
    },
    "data_sources": {
        "primary": "github",
        "fallback": "api",
        "cache_ttl": 300,  # 5 minutes
        "rate_limiting": {
            "requests_per_minute": 60,
            "burst_limit": 10
        }
    },
    "symbols": {
        "default": ["BTC/USDT", "ETH/USDT", "ADA/USDT", "DOT/USDT", "LINK/USDT"],
        "watchlist": ["BTC/USDT", "ETH/USDT"],
        "max_concurrent_analysis": 10
    },
    "analysis": {
        "multi_timeframe": {
            "enabled": True,
            "timeframes": ["1h", "4h", "1d", "1w"],
            "primary_trend_weight": 0.4
        },
        "spiking_transformer": {
            "enabled": True,
            "d_model": 64,
            "n_heads": 4,
            "seq_len": 10
        },
        "ai_models": {
            "regime_classifier": True,
            "pattern_predictor": True,
            "confidence_threshold": 0.6
        }
    },
    "trading": {
        "risk_management": {
            "total_capital": 10000,
            "max_risk_per_trade": 0.02,  # 2%
            "max_portfolio_risk": 0.10,  # 10%
            "position_sizing": "dynamic"
        },
        "signals": {
            "min_confidence": 0.7,
            "max_signals_per_day": 5,
            "cooldown_period_minutes": 30
        }
    },
    "api": {
        "host": "0.0.0.0",
        "port": 8000,
        "cors_origins": ["http://localhost:3000"],
        "rate_limiting": {
            "requests_per_minute": 100,
            "burst_limit": 20
        }
    },
    "monitoring": {
        "health_check_interval": 60,
        "performance_tracking": True,
        "memory_monitoring": True,
        "alert_channels": ["log", "console"]
    },
    "backtesting": {
        "initial_capital": 10000,
        "commission": 0.001,
        "slippage": 0.0005,
        "default_window_size": 1000,
        "default_step_size": 200
    }
}
