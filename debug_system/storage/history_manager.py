import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import threading

logger = logging.getLogger(__name__)

class HistoryManager:
    def __init__(self, db_path: str = "./debug_history.db"):
        self.db_path = Path(db_path)
        self._init_database()
        
    def _init_database(self):
        """مقداردهی اولیه دیتابیس"""
        conn = self._get_connection()
        try:
            # جدول تاریخچه اندپوینت‌ها
            conn.execute('''
                CREATE TABLE IF NOT EXISTS endpoint_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    endpoint TEXT NOT NULL,
                    method TEXT NOT NULL,
                    response_time REAL NOT NULL,
                    status_code INTEGER NOT NULL,
                    cache_used BOOLEAN NOT NULL,
                    api_calls INTEGER NOT NULL,
                    memory_used REAL NOT NULL,
                    cpu_impact REAL NOT NULL,
                    timestamp DATETIME NOT NULL,
                    params TEXT
                )
            ''')
            
            # جدول تاریخچه متریک‌های سیستم
            conn.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cpu_percent REAL NOT NULL,
                    memory_percent REAL NOT NULL,
                    disk_usage REAL NOT NULL,
                    network_sent_mb REAL NOT NULL,
                    network_recv_mb REAL NOT NULL,
                    active_connections INTEGER NOT NULL,
                    timestamp DATETIME NOT NULL
                )
            ''')
            
            # جدول تاریخچه هشدارها
            conn.execute('''
                CREATE TABLE IF NOT EXISTS alert_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    level TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    source TEXT NOT NULL,
                    acknowledged BOOLEAN NOT NULL,
                    timestamp DATETIME NOT NULL,
                    data TEXT
                )
            ''')
            
            # ایندکس‌ها برای performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_endpoint_timestamp ON endpoint_history(endpoint, timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON system_metrics_history(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_alert_timestamp ON alert_history(timestamp)')
            
            conn.commit()
            logger.info("✅ History database initialized")
            
        except Exception as e:
            logger.error(f"❌ Database initialization error: {e}")
        finally:
            conn.close()
    
    def _get_connection(self) -> sqlite3.Connection:
        """دریافت connection به دیتابیس"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def save_endpoint_call(self, endpoint_data: Dict[str, Any]):
        """ذخیره فراخوانی اندپوینت در تاریخچه"""
        conn = self._get_connection()
        try:
            conn.execute('''
                INSERT INTO endpoint_history 
                (endpoint, method, response_time, status_code, cache_used, api_calls, memory_used, cpu_impact, timestamp, params)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                endpoint_data['endpoint'],
                endpoint_data['method'],
                endpoint_data['response_time'],
                endpoint_data['status_code'],
                endpoint_data['cache_used'],
                endpoint_data['api_calls'],
                endpoint_data['memory_used'],
                endpoint_data['cpu_impact'],
                endpoint_data['timestamp'],
                json.dumps(endpoint_data['params']) if endpoint_data.get('params') else None
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"❌ Error saving endpoint call: {e}")
        finally:
            conn.close()
    
    def save_system_metrics(self, metrics_data: Dict[str, Any]):
        """ذخیره متریک‌های سیستم در تاریخچه"""
        conn = self._get_connection()
        try:
            conn.execute('''
                INSERT INTO system_metrics_history 
                (cpu_percent, memory_percent, disk_usage, network_sent_mb, network_recv_mb, active_connections, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics_data['cpu_percent'],
                metrics_data['memory_percent'],
                metrics_data['disk_usage'],
                metrics_data['network_sent_mb_sec'],
                metrics_data['network_recv_mb_sec'],
                metrics_data['active_connections'],
                metrics_data['timestamp']
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"❌ Error saving system metrics: {e}")
        finally:
            conn.close()
    
    def save_alert(self, alert_data: Dict[str, Any]):
        """ذخیره هشدار در تاریخچه"""
        conn = self._get_connection()
        try:
            conn.execute('''
                INSERT INTO alert_history 
                (level, alert_type, title, message, source, acknowledged, timestamp, data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert_data['level'],
                alert_data['type'],
                alert_data['title'],
                alert_data['message'],
                alert_data['source'],
                alert_data.get('acknowledged', False),
                alert_data['timestamp'],
                json.dumps(alert_data['data']) if alert_data.get('data') else None
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"❌ Error saving alert: {e}")
        finally:
            conn.close()
    
    def get_endpoint_history(self, 
                           endpoint: str = None,
                           start_date: datetime = None,
                           end_date: datetime = None,
                           limit: int = 1000) -> List[Dict[str, Any]]:
        """دریافت تاریخچه اندپوینت"""
        conn = self._get_connection()
        try:
            query = 'SELECT * FROM endpoint_history WHERE 1=1'
            params = []
            
            if endpoint:
                query += ' AND endpoint = ?'
                params.append(endpoint)
            
            if start_date:
                query += ' AND timestamp >= ?'
                params.append(start_date.isoformat())
            
            if end_date:
                query += ' AND timestamp <= ?'
                params.append(end_date.isoformat())
            
            query += ' ORDER BY timestamp DESC LIMIT ?'
            params.append(limit)
            
            cursor = conn.execute(query, params)
            results = []
            
            for row in cursor:
                results.append({
                    'endpoint': row['endpoint'],
                    'method': row['method'],
                    'response_time': row['response_time'],
                    'status_code': row['status_code'],
                    'cache_used': bool(row['cache_used']),
                    'api_calls': row['api_calls'],
                    'memory_used': row['memory_used'],
                    'cpu_impact': row['cpu_impact'],
                    'timestamp': row['timestamp'],
                    'params': json.loads(row['params']) if row['params'] else {}
                })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error getting endpoint history: {e}")
            return []
        finally:
            conn.close()
    
    def get_system_metrics_history(self,
                                 start_date: datetime = None,
                                 end_date: datetime = None,
                                 limit: int = 1000) -> List[Dict[str, Any]]:
        """دریافت تاریخچه متریک‌های سیستم"""
        conn = self._get_connection()
        try:
            query = 'SELECT * FROM system_metrics_history WHERE 1=1'
            params = []
            
            if start_date:
                query += ' AND timestamp >= ?'
                params.append(start_date.isoformat())
            
            if end_date:
                query += ' AND timestamp <= ?'
                params.append(end_date.isoformat())
            
            query += ' ORDER BY timestamp DESC LIMIT ?'
            params.append(limit)
            
            cursor = conn.execute(query, params)
            results = []
            
            for row in cursor:
                results.append({
                    'cpu_percent': row['cpu_percent'],
                    'memory_percent': row['memory_percent'],
                    'disk_usage': row['disk_usage'],
                    'network_sent_mb': row['network_sent_mb'],
                    'network_recv_mb': row['network_recv_mb'],
                    'active_connections': row['active_connections'],
                    'timestamp': row['timestamp']
                })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error getting system metrics history: {e}")
            return []
        finally:
            conn.close()
    
    def get_alert_history(self,
                        level: str = None,
                        start_date: datetime = None,
                        end_date: datetime = None,
                        limit: int = 1000) -> List[Dict[str, Any]]:
        """دریافت تاریخچه هشدارها"""
        conn = self._get_connection()
        try:
            query = 'SELECT * FROM alert_history WHERE 1=1'
            params = []
            
            if level:
                query += ' AND level = ?'
                params.append(level)
            
            if start_date:
                query += ' AND timestamp >= ?'
                params.append(start_date.isoformat())
            
            if end_date:
                query += ' AND timestamp <= ?'
                params.append(end_date.isoformat())
            
            query += ' ORDER BY timestamp DESC LIMIT ?'
            params.append(limit)
            
            cursor = conn.execute(query, params)
            results = []
            
            for row in cursor:
                results.append({
                    'level': row['level'],
                    'type': row['alert_type'],
                    'title': row['title'],
                    'message': row['message'],
                    'source': row['source'],
                    'acknowledged': bool(row['acknowledged']),
                    'timestamp': row['timestamp'],
                    'data': json.loads(row['data']) if row['data'] else {}
                })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error getting alert history: {e}")
            return []
        finally:
            conn.close()
    
    def get_performance_trends(self, days: int = 30) -> Dict[str, Any]:
        """دریافت روندهای عملکرد"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        conn = self._get_connection()
        try:
            # روند زمان پاسخ
            cursor = conn.execute('''
                SELECT 
                    DATE(timestamp) as date,
                    AVG(response_time) as avg_response_time,
                    COUNT(*) as call_count
                FROM endpoint_history 
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY DATE(timestamp)
                ORDER BY date
            ''', (start_date.isoformat(), end_date.isoformat()))
            
            response_trends = []
            for row in cursor:
                response_trends.append({
                    'date': row['date'],
                    'avg_response_time': row['avg_response_time'],
                    'call_count': row['call_count']
                })
            
            # روند استفاده از منابع
            cursor = conn.execute('''
                SELECT 
                    DATE(timestamp) as date,
                    AVG(cpu_percent) as avg_cpu,
                    AVG(memory_percent) as avg_memory,
                    AVG(disk_usage) as avg_disk
                FROM system_metrics_history 
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY DATE(timestamp)
                ORDER BY date
            ''', (start_date.isoformat(), end_date.isoformat()))
            
            resource_trends = []
            for row in cursor:
                resource_trends.append({
                    'date': row['date'],
                    'avg_cpu': row['avg_cpu'],
                    'avg_memory': row['avg_memory'],
                    'avg_disk': row['avg_disk']
                })
            
            return {
                'response_trends': response_trends,
                'resource_trends': resource_trends,
                'time_period_days': days,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting performance trends: {e}")
            return {}
        finally:
            conn.close()
    
    def cleanup_old_data(self, days: int = 90):
        """پاک‌سازی داده‌های قدیمی"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        conn = self._get_connection()
        try:
            # پاک‌سازی داده‌های قدیمی
            conn.execute('DELETE FROM endpoint_history WHERE timestamp < ?', 
                       (cutoff_date.isoformat(),))
            conn.execute('DELETE FROM system_metrics_history WHERE timestamp < ?', 
                       (cutoff_date.isoformat(),))
            conn.execute('DELETE FROM alert_history WHERE timestamp < ?', 
                       (cutoff_date.isoformat(),))
            
            conn.commit()
            
            # vacuum برای آزادسازی فضای دیتابیس
            conn.execute('VACUUM')
            conn.commit()
            
            logger.info(f"🧹 Cleaned up data older than {days} days")
            
        except Exception as e:
            logger.error(f"❌ Error cleaning up old data: {e}")
        finally:
            conn.close()

# ایجاد نمونه گلوبال
history_manager = HistoryManager()
