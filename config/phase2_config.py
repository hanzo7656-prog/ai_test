# 📁 config/phase2_config.py

PHASE2_CONFIG = {
    "signal_engine": {
        "weights": {
            "momentum": 0.25,
            "trend": 0.20, 
            "volume": 0.15,
            "volatility": 0.15,
            "pattern": 0.15,
            "risk": 0.10
        },
        "confidence_thresholds": {
            "strong_buy": 0.6,
            "buy": 0.2,
            "sell": -0.2,
            "strong_sell": -0.6
        }
    },
    "risk_management": {
        "total_capital": 10000,
        "max_risk_per_trade": 0.02,  # 2%
        "portfolio_heat_limit": 0.1,  # 10%
        "volatility_adjustment": True,
        "correlation_penalty": True,
        "max_leverage": 5.0,
        "min_leverage": 1.0
    },
    "position_sizing": {
        "min_position_size": 0.001,
        "max_position_size": 0.1,  # 10% of capital
        "volatility_thresholds": {
            "high_volatility": 5.0,  # ATR percentage
            "low_volatility": 1.0
        }
    }
}
