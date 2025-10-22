# 📁 config/phase1_config.py

PHASE1_CONFIG = {
    "spiking_transformer": {
        "d_model": 64,
        "n_heads": 4,
        "num_layers": 2,
        "seq_len": 10,
        "spike_threshold": 1.0,
        "membrane_decay": 0.9
    },
    "technical_analysis": {
        "rsi_periods": [14, 21, 28],
        "macd_params": {"fast": 12, "slow": 26, "signal": 9},
        "bollinger_params": {"period": 20, "std": 2},
        "timeframes": ["1h", "4h", "1d"]
    },
    "pattern_recognition": {
        "enabled_patterns": ["doji", "hammer", "engulfing", "evening_star", "morning_star"]
    }
}
