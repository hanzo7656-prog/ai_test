# database_manager.py - مدیریت پایگاه داده در ریپو
import sqlite3
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class TradingDatabase:
    """پایگاه داده تریدینگ در ریپو"""
    
    def __init__(self, db_path: str = "./trading_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """ایجاد جداول پایگاه داده"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # جدول داده‌های قیمتی
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            )
        ''')
        
        # جدول اندیکاتورهای تکنیکال
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS technical_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                rsi REAL,
                macd REAL,
                macd_signal REAL,
                macd_histogram REAL,
                bollinger_upper REAL,
                bollinger_middle REAL,
                bollinger_lower REAL,
                sma_20 REAL,
                ema_12 REAL,
                atr REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            )
        ''')
        
        # جدول سیگنال‌های AI
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                signal_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                price REAL NOT NULL,
                model_version TEXT NOT NULL,
                features_json TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # جدول معاملات
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                entry_time DATETIME NOT NULL,
                exit_time DATETIME,
                entry_price REAL NOT NULL,
                exit_price REAL,
                position_type TEXT NOT NULL,
                size REAL NOT NULL,
                pnl REAL,
                status TEXT DEFAULT 'open',
                signal_id INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("✅ پایگاه داده تریدینگ راه‌اندازی شد")
    
    def save_price_data(self, symbol: str, data: List[Dict]):
        """ذخیره داده‌های قیمتی"""
        conn = sqlite3.connect(self.db_path)
        
        for item in data:
            conn.execute('''
                INSERT OR REPLACE INTO price_data 
                (symbol, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                item['timestamp'],
                item['open'],
                item['high'],
                item['low'],
                item['close'],
                item['volume']
            ))
        
        conn.commit()
        conn.close()
        logger.info(f"✅ داده‌های قیمتی {symbol} ذخیره شد")
    
    def save_technical_indicators(self, symbol: str, indicators: Dict):
        """ذخیره اندیکاتورهای تکنیکال"""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute('''
            INSERT OR REPLACE INTO technical_indicators 
            (symbol, timestamp, rsi, macd, macd_signal, macd_histogram,
             bollinger_upper, bollinger_middle, bollinger_lower,
             sma_20, ema_12, atr)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol,
            indicators['timestamp'],
            indicators.get('rsi'),
            indicators.get('macd'),
            indicators.get('macd_signal'),
            indicators.get('macd_histogram'),
            indicators.get('bollinger_upper'),
            indicators.get('bollinger_middle'),
            indicators.get('bollinger_lower'),
            indicators.get('sma_20'),
            indicators.get('ema_12'),
            indicators.get('atr')
        ))
        
        conn.commit()
        conn.close()
    
    def get_historical_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """دریافت داده‌های تاریخی"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT p.timestamp, p.open, p.high, p.low, p.close, p.volume,
                   t.rsi, t.macd, t.macd_signal, t.macd_histogram,
                   t.bollinger_upper, t.bollinger_middle, t.bollinger_lower,
                   t.sma_20, t.ema_12, t.atr
            FROM price_data p
            LEFT JOIN technical_indicators t ON p.symbol = t.symbol AND p.timestamp = t.timestamp
            WHERE p.symbol = ? AND p.timestamp >= datetime('now', ?)
            ORDER BY p.timestamp
        '''
        
        df = pd.read_sql_query(query, conn, params=[symbol, f'-{days} days'])
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        return df
    
    def save_ai_signal(self, signal_data: Dict):
        """ذخیره سیگنال AI"""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute('''
            INSERT INTO ai_signals 
            (symbol, timestamp, signal_type, confidence, price, model_version, features_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal_data['symbol'],
            signal_data['timestamp'],
            signal_data['signal_type'],
            signal_data['confidence'],
            signal_data['price'],
            signal_data['model_version'],
            json.dumps(signal_data.get('features', {}))
        ))
        
        conn.commit()
        conn.close()

# ایجاد نمونه گلوبال
trading_db = TradingDatabase()
