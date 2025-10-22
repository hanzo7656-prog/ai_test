# 📁 config/phase4_config.py

PHASE4_CONFIG = {
    "backtesting": {
        "initial_capital": 10000,
        "commission": 0.001,  # 0.1%
        "slippage": 0.0005,   # 0.05%
        "walk_forward": {
            "minimum_window_size": 100,
            "default_window_size": 1000,
            "default_step_size": 200,
            "min_test_period": 10
        },
        "monte_carlo": {
            "default_simulations": 10000,
            "confidence_level": 0.95,
            "ruin_threshold": 0.5  # 50% loss
        }
    },
    "visualization": {
        "theme": {
            "background": "#0E1117",
            "text": "#FFFFFF", 
            "grid": "#1E2130",
            "up": "#00C853",
            "down": "#FF1744",
            "neutral": "#FFC107"
        },
        "dashboard": {
            "update_interval": 60,  # seconds
            "max_display_points": 1000,
            "default_height": 800
        },
        "charts": {
            "candlestick": True,
            "volume": True,
            "indicators": ["RSI", "MACD", "BBANDS"],
            "drawing_tools": True
        }
    },
    "performance_metrics": {
        "primary_metrics": ["total_return", "sharpe_ratio", "max_drawdown", "win_rate"],
        "secondary_metrics": ["profit_factor", "avg_trade", "best_trade", "worst_trade"],
        "risk_metrics": ["var", "expected_shortfall", "probability_of_ruin"]
    },
    "reporting": {
        "auto_generate_reports": True,
        "report_formats": ["html", "pdf"],
        "email_alerts": True,
        "performance_thresholds": {
            "min_win_rate": 40,
            "max_drawdown": 20,
            "min_sharpe_ratio": 0.5
        }
    }
}
