import logging
import smtplib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import asyncio
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING" 
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class AlertType(Enum):
    PERFORMANCE = "PERFORMANCE"
    ERROR = "ERROR"
    SECURITY = "SECURITY"
    SYSTEM = "SYSTEM"
    API = "API"
    NETWORK = "NETWORK"
    DATABASE = "DATABASE"

class AlertManager:
    def __init__(self):
        self.active_alerts = []
        self.alert_history = deque(maxlen=10000)
        self.alert_rules = self._initialize_alert_rules()
        self.notification_channels = {}
        self.alert_counters = defaultdict(int)
        self.alert_cooldowns = {}
        
        # تنظیمات پیش‌فرض
        self.alert_settings = {
            'email_enabled': False,
            'slack_enabled': False,
            'webhook_enabled': False,
            'console_enabled': True,
            'cooldown_minutes': {
                'INFO': 5,
                'WARNING': 10,
                'ERROR': 15,
                'CRITICAL': 30
            },
            'retention_days': 90
        }
        
        logger.info("✅ Alert Manager initialized")

    def create_alert(self, 
                    level: AlertLevel,
                    alert_type: AlertType,
                    title: str,
                    message: str,
                    source: str,
                    data: Dict[str, Any] = None,
                    auto_acknowledge: bool = False) -> Dict[str, Any]:
        """ایجاد هشدار جدید"""
        
        # بررسی cooldown برای جلوگیری از اسپم
        if self._is_in_cooldown(source, level, alert_type):
            logger.debug(f"🔇 Alert cooldown active for {source} - {alert_type.value}")
            return None
        
        alert_id = len(self.alert_history) + 1
        
        alert = {
            'id': alert_id,
            'level': level.value,
            'type': alert_type.value,
            'title': title,
            'message': message,
            'source': source,
            'timestamp': datetime.now().isoformat(),
            'data': data or {},
            'acknowledged': auto_acknowledge,
            'notified': False,
            'resolved': False,
            'resolved_at': None,
            'resolved_by': None
        }
        
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        
        # آپدیت شمارنده
        alert_key = f"{source}_{alert_type.value}_{level.value}"
        self.alert_counters[alert_key] += 1
        
        # آپدیت cooldown
        self._update_cooldown(source, level, alert_type)
        
        # ارسال نوتیفیکیشن
        asyncio.create_task(self._send_notifications(alert))
        
        logger.warning(f"🚨 {level.value} Alert: {title} - {message}")
        
        return alert

    def acknowledge_alert(self, alert_id: int, user: str = "system") -> bool:
        """تأیید هشدار"""
        for alert in self.active_alerts:
            if alert['id'] == alert_id:
                alert['acknowledged'] = True
                alert['acknowledged_by'] = user
                alert['acknowledged_at'] = datetime.now().isoformat()
                
                logger.info(f"✅ Alert {alert_id} acknowledged by {user}")
                return True
        
        return False

    def resolve_alert(self, alert_id: int, resolved_by: str = "system", 
                     resolution_notes: str = "") -> bool:
        """حل هشدار"""
        for alert in self.active_alerts:
            if alert['id'] == alert_id:
                alert['resolved'] = True
                alert['resolved_at'] = datetime.now().isoformat()
                alert['resolved_by'] = resolved_by
                alert['resolution_notes'] = resolution_notes
                
                # حذف از active alerts
                self.active_alerts.remove(alert)
                
                logger.info(f"✅ Alert {alert_id} resolved by {resolved_by}")
                return True
        
        return False

    def get_active_alerts(self, 
                         level: AlertLevel = None, 
                         alert_type: AlertType = None,
                         source: str = None) -> List[Dict[str, Any]]:
        """دریافت هشدارهای فعال"""
        filtered_alerts = self.active_alerts
        
        if level:
            filtered_alerts = [a for a in filtered_alerts if a['level'] == level.value]
        
        if alert_type:
            filtered_alerts = [a for a in filtered_alerts if a['type'] == alert_type.value]
        
        if source:
            filtered_alerts = [a for a in filtered_alerts if a['source'] == source]
        
        return filtered_alerts

    def get_alert_history(self,
                         level: AlertLevel = None,
                         alert_type: AlertType = None,
                         source: str = None,
                         start_date: datetime = None,
                         end_date: datetime = None,
                         limit: int = 1000) -> List[Dict[str, Any]]:
        """دریافت تاریخچه هشدارها"""
        filtered_alerts = list(self.alert_history)
        
        # فیلتر بر اساس سطح
        if level:
            filtered_alerts = [a for a in filtered_alerts if a['level'] == level.value]
        
        # فیلتر بر اساس نوع
        if alert_type:
            filtered_alerts = [a for a in filtered_alerts if a['type'] == alert_type.value]
        
        # فیلتر بر اساس منبع
        if source:
            filtered_alerts = [a for a in filtered_alerts if a['source'] == source]
        
        # فیلتر بر اساس تاریخ
        if start_date:
            filtered_alerts = [
                a for a in filtered_alerts 
                if datetime.fromisoformat(a['timestamp']) >= start_date
            ]
        
        if end_date:
            filtered_alerts = [
                a for a in filtered_alerts 
                if datetime.fromisoformat(a['timestamp']) <= end_date
            ]
        
        # مرتب‌سازی بر اساس تاریخ (جدیدترین اول)
        filtered_alerts.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return filtered_alerts[:limit]

    def get_alert_stats(self, hours: int = 24) -> Dict[str, Any]:
        """دریافت آمار هشدارها"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_alerts = [
            alert for alert in self.alert_history 
            if datetime.fromisoformat(alert['timestamp']) >= cutoff_time
        ]
        
        stats = {
            'total_alerts': len(recent_alerts),
            'active_alerts': len(self.active_alerts),
            'resolved_alerts': len([a for a in recent_alerts if a.get('resolved', False)]),
            'by_level': defaultdict(int),
            'by_type': defaultdict(int),
            'by_source': defaultdict(int),
            'time_period_hours': hours
        }
        
        for alert in recent_alerts:
            stats['by_level'][alert['level']] += 1
            stats['by_type'][alert['type']] += 1
            stats['by_source'][alert['source']] += 1
        
        return stats

    def add_notification_channel(self, channel_type: str, config: Dict[str, Any]):
        """اضافه کردن کانال نوتیفیکیشن"""
        self.notification_channels[channel_type] = config
        
        if channel_type == 'email':
            self.alert_settings['email_enabled'] = True
        elif channel_type == 'slack':
            self.alert_settings['slack_enabled'] = True
        elif channel_type == 'webhook':
            self.alert_settings['webhook_enabled'] = True
        
        logger.info(f"✅ Added {channel_type} notification channel")

    def _send_notifications(self, alert: Dict[str, Any]):
        """ارسال نوتیفیکیشن برای هشدار"""
    
        # فقط برای هشدارهای ERROR و CRITICAL نوتیفیکیشن بفرست
        if alert['level'] in [AlertLevel.INFO.value, AlertLevel.WARNING.value]:
            return
    
        try:
            # ایمیل
            if self.alert_settings['email_enabled'] and 'email' in self.notification_channels:
                # اجرای غیرهمزمان برای ایمیل
                import asyncio
                try:
                    asyncio.create_task(self._send_email_alert(alert))
                except:
                    # اگر create_task شکست خورد، به صورت sync اجرا کن
                    import threading
                    thread = threading.Thread(target=lambda: asyncio.run(self._send_email_alert(alert)))
                    thread.daemon = True
                    thread.start()
        
            # Slack
            if self.alert_settings['slack_enabled'] and 'slack' in self.notification_channels:
                try:
                    asyncio.create_task(self._send_slack_alert(alert))
                except:
                    thread = threading.Thread(target=lambda: asyncio.run(self._send_slack_alert(alert)))
                    thread.daemon = True
                    thread.start()
        
            # Webhook
            if self.alert_settings['webhook_enabled'] and 'webhook' in self.notification_channels:
                try:
                    asyncio.create_task(self._send_webhook_alert(alert))
                except:
                    thread = threading.Thread(target=lambda: asyncio.run(self._send_webhook_alert(alert)))
                    thread.daemon = True
                    thread.start()
        
            # Console (همیشه فعال)
            if self.alert_settings['console_enabled']:
                self._send_console_alert(alert)
        
            alert['notified'] = True
          
        except Exception as e:
            logger.error(f"❌ Error sending alert notifications: {e}")

    async def _send_email_alert(self, alert: Dict[str, Any]):
        """ارسال هشدار از طریق ایمیل"""
        try:
            config = self.notification_channels['email']
            
            # ایجاد پیام
            msg = MIMEMultipart()
            msg['From'] = config['from_email']
            msg['To'] = ', '.join(config['to_emails'])
            msg['Subject'] = f"🚨 {alert['level']} Alert: {alert['title']}"
            
            # بدنه ایمیل
            body = f"""
            VortexAI Alert System
            ====================
            
            Alert Details:
            -------------
            Level: {alert['level']}
            Type: {alert['type']}
            Source: {alert['source']}
            Time: {alert['timestamp']}
            
            Message:
            {alert['message']}
            
            Additional Data:
            {json.dumps(alert['data'], indent=2, ensure_ascii=False)}
            
            ---
            This is an automated message from VortexAI Monitoring System.
            """
            
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            # ارسال ایمیل
            with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
                server.starttls()
                server.login(config['username'], config['password'])
                server.send_message(msg)
            
            logger.info(f"📧 Email alert sent for: {alert['title']}")
            
        except Exception as e:
            logger.error(f"❌ Email alert failed: {e}")

    async def _send_slack_alert(self, alert: Dict[str, Any]):
        """ارسال هشدار به Slack"""
        try:
            config = self.notification_channels['slack']
            
            # ایجاد payload برای Slack
            slack_payload = {
                'text': f"🚨 {alert['level']} Alert: {alert['title']}",
                'blocks': [
                    {
                        'type': 'header',
                        'text': {
                            'type': 'plain_text',
                            'text': f"{alert['level']} Alert: {alert['title']}"
                        }
                    },
                    {
                        'type': 'section',
                        'fields': [
                            {
                                'type': 'mrkdwn',
                                'text': f"*Type:*\n{alert['type']}"
                            },
                            {
                                'type': 'mrkdwn', 
                                'text': f"*Source:*\n{alert['source']}"
                            },
                            {
                                'type': 'mrkdwn',
                                'text': f"*Time:*\n{alert['timestamp']}"
                            }
                        ]
                    },
                    {
                        'type': 'section',
                        'text': {
                            'type': 'mrkdwn',
                            'text': f"*Message:*\n{alert['message']}"
                        }
                    }
                ]
            }
            
            # در یک پیاده‌سازی واقعی، اینجا درخواست HTTP به Slack می‌فرستیم
            logger.info(f"💬 Slack alert prepared for: {alert['title']}")
            # await self._send_slack_webhook(config['webhook_url'], slack_payload)
            
        except Exception as e:
            logger.error(f"❌ Slack alert failed: {e}")

    async def _send_webhook_alert(self, alert: Dict[str, Any]):
        """ارسال هشدار به Webhook"""
        try:
            config = self.notification_channels['webhook']
            
            # ایجاد payload برای webhook
            webhook_payload = {
                'event': 'alert',
                'alert': alert,
                'sent_at': datetime.now().isoformat()
            }
            
            # در یک پیاده‌سازی واقعی، اینجا درخواست HTTP به webhook می‌فرستیم
            logger.info(f"🌐 Webhook alert prepared for: {alert['title']}")
            # await self._send_http_request(config['url'], webhook_payload)
            
        except Exception as e:
            logger.error(f"❌ Webhook alert failed: {e}")

    def _send_console_alert(self, alert: Dict[str, Any]):
        """ارسال هشدار به کنسول"""
        try:
            # رنگ‌بندی بر اساس سطح هشدار
            color_codes = {
                'INFO': '\033[94m',      # آبی
                'WARNING': '\033[93m',   # زرد  
                'ERROR': '\033[91m',     # قرمز
                'CRITICAL': '\033[41m'   # پس‌زمینه قرمز
            }
            
            reset_code = '\033[0m'
            color = color_codes.get(alert['level'], '\033[0m')
            
            console_message = f"""
{color}╔═══════════════════════════════════════════════════════════════╗
║                    🚨 VORTEXAI ALERT                    ║
╠═══════════════════════════════════════════════════════════════╣
║ Level: {alert['level']:<15} Type: {alert['type']:<20} ║
║ Source: {alert['source']:<50} ║
║ Time: {alert['timestamp']:<45} ║
╠═══════════════════════════════════════════════════════════════╣
║ {alert['message']:<63} ║
╚═══════════════════════════════════════════════════════════════╝{reset_code}
            """
            
            print(console_message)
            logger.info(f"📟 Console alert displayed for: {alert['title']}")
            
        except Exception as e:
            logger.error(f"❌ Console alert failed: {e}")

    def _initialize_alert_rules(self) -> Dict[str, Any]:
        """مقداردهی اولیه قوانین هشدار"""
        return {
            'performance': {
                'high_cpu': {
                    'threshold': 90, 
                    'level': AlertLevel.CRITICAL,
                    'message': 'CPU usage exceeded threshold'
                },
                'high_memory': {
                    'threshold': 90, 
                    'level': AlertLevel.CRITICAL,
                    'message': 'Memory usage exceeded threshold'
                },
                'slow_response': {
                    'threshold': 3.0, 
                    'level': AlertLevel.WARNING,
                    'message': 'Response time exceeded threshold'
                }
            },
            'errors': {
                'api_timeout': {
                    'level': AlertLevel.ERROR,
                    'message': 'API request timeout'
                },
                'database_error': {
                    'level': AlertLevel.CRITICAL,
                    'message': 'Database connection error'
                },
                'external_api_error': {
                    'level': AlertLevel.WARNING,
                    'message': 'External API error'
                }
            },
            'security': {
                'rate_limit_exceeded': {
                    'level': AlertLevel.WARNING,
                    'message': 'Rate limit exceeded'
                },
                'suspicious_activity': {
                    'level': AlertLevel.CRITICAL, 
                    'message': 'Suspicious activity detected'
                },
                'failed_authentication': {
                    'level': AlertLevel.ERROR,
                    'message': 'Multiple authentication failures'
                }
            }
        }

    def _is_in_cooldown(self, source: str, level: AlertLevel, alert_type: AlertType) -> bool:
        """بررسی cooldown برای هشدار"""
        cooldown_key = f"{source}_{alert_type.value}_{level.value}"
        cooldown_minutes = self.alert_settings['cooldown_minutes'].get(level.value, 5)
        
        if cooldown_key in self.alert_cooldowns:
            last_alert_time = self.alert_cooldowns[cooldown_key]
            time_since_last = datetime.now() - last_alert_time
            
            if time_since_last < timedelta(minutes=cooldown_minutes):
                return True
        
        return False

    def _update_cooldown(self, source: str, level: AlertLevel, alert_type: AlertType):
        """آپدیت زمان cooldown"""
        cooldown_key = f"{source}_{alert_type.value}_{level.value}"
        self.alert_cooldowns[cooldown_key] = datetime.now()

    def auto_resolve_alerts(self, source: str = None):
        """حل خودکار هشدارهای قدیمی"""
        resolved_count = 0
        current_time = datetime.now()
        
        for alert in self.active_alerts[:]:  # کپی از لیست
            # فقط هشدارهای INFO و WARNING بعد از ۱ ساعت به صورت خودکار حل می‌شوند
            if alert['level'] in ['INFO', 'WARNING']:
                alert_time = datetime.fromisoformat(alert['timestamp'])
                time_since_alert = current_time - alert_time
                
                if time_since_alert > timedelta(hours=1):
                    if source is None or alert['source'] == source:
                        self.resolve_alert(alert['id'], 'auto_resolver', 
                                         'Automatically resolved after 1 hour')
                        resolved_count += 1
        
        if resolved_count > 0:
            logger.info(f"🧹 Auto-resolved {resolved_count} alerts")

    def cleanup_old_alerts(self):
        """پاک‌سازی هشدارهای قدیمی"""
        cutoff_time = datetime.now() - timedelta(days=self.alert_settings['retention_days'])
        
        # پاک‌سازی از تاریخچه
        self.alert_history = deque([
            alert for alert in self.alert_history 
            if datetime.fromisoformat(alert['timestamp']) >= cutoff_time
        ], maxlen=10000)
        
        # پاک‌سازی از active alerts (فقط اگر خیلی قدیمی باشند)
        for alert in self.active_alerts[:]:
            alert_time = datetime.fromisoformat(alert['timestamp'])
            if alert_time < cutoff_time - timedelta(days=7):  # 7 روز اضافی برای active alerts
                self.active_alerts.remove(alert)
        
        logger.info(f"🧹 Cleaned up alerts older than {self.alert_settings['retention_days']} days")

    def get_alert_trends(self, days: int = 30) -> Dict[str, Any]:
        """دریافت روند هشدارها"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        alerts_in_period = [
            alert for alert in self.alert_history
            if start_date <= datetime.fromisoformat(alert['timestamp']) <= end_date
        ]
        
        # گروه‌بندی بر اساس روز
        daily_trends = defaultdict(lambda: {
            'total': 0,
            'by_level': defaultdict(int),
            'by_type': defaultdict(int)
        })
        
        for alert in alerts_in_period:
            alert_date = datetime.fromisoformat(alert['timestamp']).strftime('%Y-%m-%d')
            daily_trends[alert_date]['total'] += 1
            daily_trends[alert_date]['by_level'][alert['level']] += 1
            daily_trends[alert_date]['by_type'][alert['type']] += 1
        
        return {
            'period_days': days,
            'total_alerts': len(alerts_in_period),
            'daily_trends': dict(daily_trends),
            'timestamp': datetime.now().isoformat()
        }

# ایجاد نمونه گلوبال
alert_manager = AlertManager()
