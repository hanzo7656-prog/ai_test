# 📁 config/phase3_config.py

PHASE3_CONFIG = {
    "multi_timeframe": {
        "timeframe_hierarchy": ["1w", "1d", "4h", "1h", "15m"],
        "analysis_weights": {
            "primary_trend": 0.4,
            "medium_trend": 0.3, 
            "short_trend": 0.2,
            "intraday": 0.1
        },
        "minimum_data_points": {
            "1w": 52,   # 1 year
            "1d": 90,   # 3 months
            "4h": 120,  # 20 days
            "1h": 168,  # 1 week
            "15m": 96   # 1 day
        }
    },
    "ai_models": {
        "regime_classifier": {
            "model_type": "RandomForest",
            "n_estimators": 100,
            "max_depth": 10,
            "retraining_interval": "30d",
            "confidence_threshold": 0.6
        },
        "pattern_predictor": {
            "model_type": "LSTM",
            "sequence_length": 20,
            "hidden_size": 50,
            "num_layers": 2,
            "prediction_horizon": "5_periods",
            "retraining_interval": "14d"
        }
    },
    "market_regimes": {
        "bull_normal": {
            "suggested_strategy": "Trend Following",
            "risk_level": "MEDIUM",
            "position_sizing": "NORMAL"
        },
        "bull_accelerating": {
            "suggested_strategy": "Momentum Trading", 
            "risk_level": "HIGH",
            "position_sizing": "REDUCED"
        },
        "bear_normal": {
            "suggested_strategy": "Short Selling",
            "risk_level": "MEDIUM", 
            "position_sizing": "NORMAL"
        },
        "bear_accelerating": {
            "suggested_strategy": "Risk Avoidance",
            "risk_level": "VERY_HIGH",
            "position_sizing": "MINIMAL"
        },
        "sideways_low_vol": {
            "suggested_strategy": "Range Trading",
            "risk_level": "LOW",
            "position_sizing": "NORMAL"
        },
        "sideways_high_vol": {
            "suggested_strategy": "Swing Trading", 
            "risk_level": "MEDIUM",
            "position_sizing": "REDUCED"
        },
        "volatile_breakout": {
            "suggested_strategy": "Breakout Trading",
            "risk_level": "HIGH",
            "position_sizing": "MINIMAL"
        }
    }
}
