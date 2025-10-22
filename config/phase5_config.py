# 📁 config/phase5_config.py

PHASE5_CONFIG = {
    "system": {
        "name": "Crypto Market Analyzer Pro",
        "version": "1.0.0", 
        "environment": "production",
        "log_level": "INFO",
        "analysis_interval": 300,
        "max_memory_usage_mb": 512
    },
    
    "data_sources": {
        "primary": "github",
        "fallback": "api", 
        "cache_ttl": 300,
        "rate_limiting": {
            "requests_per_minute": 60,
            "burst_limit": 10
        },
        "multi_source": {
            "enabled": true,
            "sources": ["binance", "bybit", "okx"],
            "weights": {"binance": 0.5, "bybit": 0.3, "okx": 0.2}
        }
    },
    
    "symbols": {
        "default": ["BTC/USDT", "ETH/USDT", "ADA/USDT", "DOT/USDT", "LINK/USDT"],
        "watchlist": ["BTC/USDT", "ETH/USDT"],
        "max_concurrent_analysis": 10
    },
    
    "analysis": {
        "multi_timeframe": {
            "enabled": true,
            "timeframes": ["1h", "4h", "1d", "1w"],
            "primary_trend_weight": 0.4
        },
        "spiking_transformer": {
            "enabled": true,
            "d_model": 64,
            "n_heads": 4, 
            "seq_len": 10,
            "sparsity_level": 0.9
        },
        "ai_models": {
            "regime_classifier": true,
            "pattern_predictor": true,
            "anomaly_detector": true,
            "signal_optimizer": true,
            "confidence_threshold": 0.6
        }
    },
    
    "trading": {
        "risk_management": {
            "total_capital": 10000,
            "max_risk_per_trade": 0.02,
            "max_portfolio_risk": 0.10,
            "position_sizing": "dynamic",
            "stop_loss_percentage": 0.02,
            "take_profit_ratios": [0.04, 0.06, 0.08]
        },
        "signals": {
            "min_confidence": 0.7,
            "max_signals_per_day": 5,
            "cooldown_period_minutes": 30
        }
    },
    
    "backtesting": {
        "initial_capital": 10000,
        "commission": 0.001,
        "slippage": 0.0005,
        "default_window_size": 1000,
        "default_step_size": 200,
        "monte_carlo_simulations": 5000
    },
    
    "api": {
        "host": "0.0.0.0",
        "port": 8000,
        "cors_origins": ["http://localhost:3000"],
        "rate_limiting": {
            "requests_per_minute": 100,
            "burst_limit": 20
        },
        "authentication": {
            "api_key_required": true,
            "jwt_expiry_minutes": 60
        }
    },
    
    "monitoring": {
        "health_check_interval": 60,
        "performance_tracking": true,
        "memory_monitoring": true,
        "alert_channels": ["log", "console", "email"],
        "metrics_retention_days": 30
    },
    
    "optimization": {
        "memory_optimization": true,
        "sparse_optimization": true, 
        "model_compression": true,
        "cache_optimization": true
    },
    
    "alerts": {
        "email": {
            "enabled": false,
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "from_email": "alerts@yourapp.com",
            "recipients": ["admin@yourapp.com"]
        },
        "webhook": {
            "enabled": false,
            "url": "https://hooks.slack.com/your-webhook"
        }
    }
}
