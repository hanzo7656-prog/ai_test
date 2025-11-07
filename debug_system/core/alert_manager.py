import logging
import smtplib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import asyncio

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

class AlertManager:
    def __init__(self):
        self.active_alerts = []
        self.alert_history = []
        self.alert_rules = self._initialize_alert_rules()
        self.notification_channels = {}
        self.alert_counters = defaultdict(int)
        
        # تنظیمات پیش‌فرض
        self.alert_settings = {
            'email_enabled': False,
            'slack_enabled': False,
            'webhook_enabled': False,
            'cooldown_minutes': 5
        }
    
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
        if self._is_in_cooldown(source, level):
            logger.debug(f"🔇 Alert cooldown active for {source}")
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
            'notified': False
        }
        
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        
        # آپدیت شمارنده
        self.alert_counters[f"{source}_{level.value}"] += 1
        
        # ارسال نوتیفیکیشن
        self._send_notifications(alert)
        
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
    
    def get_active_alerts(self, level: AlertLevel = None, alert_type: AlertType = None) -> List[Dict[str, Any]]:
        """دریافت هشدارهای فعال"""
        filtered_alerts = self.active_alerts
        
        if level:
            filtered_alerts = [a for a in filtered_alerts if a['level'] == level.value]
        
        if alert_type:
            filtered_alerts = [a for a in filtered_alerts if a['type'] == alert_type.value]
        
        return filtered_alerts
    
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
            'by_level': defaultdict(int),
            'by_type': defaultdict(int),
            'by_source': defaultdict(int)
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
                self._send_email_alert(alert)
            
            # Slack
            if self.alert_settings['slack_enabled'] and 'slack' in self.notification_channels:
                self._send_slack_alert(alert)
            
            # Webhook
            if self.alert_settings['webhook_enabled'] and 'webhook' in self.notification_channels:
                self._send_webhook_alert(alert)
            
            alert['notified'] = True
            
        except Exception as e:
            logger.error(f"❌ Error sending alert notifications: {e}")
    
    def _send_email_alert(self, alert: Dict[str, Any]):
        """ارسال هشدار از طریق ایمیل"""
        try:
            config = self.notification_channels['email']
            
            msg = MIMEMultipart()
            msg['From'] = config['from_email']
            msg['To'] = ', '.join(config['to_emails'])
            msg['Subject'] = f"🚨 {alert['level']} Alert: {alert['title']}"
            
            body = f"""
            Alert Details:
            -------------
            Level: {alert['level']}
            Type: {alert['type']}
            Source: {alert['source']}
            Time: {alert['timestamp']}
            
            Message:
            {alert['message']}
            
            Additional Data:
            {json.dumps(alert['data'], indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
                server.starttls()
                server.login(config['username'], config['password'])
                server.send_message(msg)
            
            logger.info(f"📧 Email alert sent for {alert['title']}")
            
        except Exception as e:
            logger.error(f"❌ Email alert failed: {e}")
    
    def _send_slack_alert(self, alert: Dict[str, Any]):
        """ارسال هشدار به Slack"""
        # پیاده‌سازی ارسال به Slack
        logger.info(f"💬 Slack alert would be sent for: {alert['title']}")
    
    def _send_webhook_alert(self, alert: Dict[str, Any]):
        """ارسال هشدار به Webhook"""
        # پیاده‌سازی ارسال Webhook
        logger.info(f"🌐 Webhook alert would be sent for: {alert['title']}")
    
    def _initialize_alert_rules(self) -> Dict[str, Any]:
        """مقداردهی اولیه قوانین هشدار"""
        return {
            'performance': {
                'high_cpu': {'threshold': 90, 'level': AlertLevel.CRITICAL},
                'high_memory': {'threshold': 90, 'level': AlertLevel.CRITICAL},
                'slow_response': {'threshold': 3.0, 'level': AlertLevel.WARNING}
            },
            'errors': {
                'api_timeout': {'level': AlertLevel.ERROR},
                'database_error': {'level': AlertLevel.CRITICAL},
                'external_api_error': {'level': AlertLevel.WARNING}
            },
            'security': {
                'rate_limit_exceeded': {'level': AlertLevel.WARNING},
                'suspicious_activity': {'level': AlertLevel.CRITICAL}
            }
        }
    
    def _is_in_cooldown(self, source: str, level: AlertLevel) -> bool:
        """بررسی cooldown برای هشدار"""
        key = f"{source}_{level.value}"
        last_alert_time = getattr(self, '_last_alert_time', {})
        
        if key in last_alert_time:
            time_since_last = datetime.now() - last_alert_time[key]
            if time_since_last < timedelta(minutes=self.alert_settings['cooldown_minutes']):
                return True
        
        self._last_alert_time = last_alert_time
        self._last_alert_time[key] = datetime.now()
        return False
    
    def clear_old_alerts(self, days: int = 30):
        """پاک کردن هشدارهای قدیمی"""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        self.alert_history = [
            alert for alert in self.alert_history 
            if datetime.fromisoformat(alert['timestamp']) >= cutoff_time
        ]
        
        # پاک کردن هشدارهای تأیید شده از active_alerts
        self.active_alerts = [alert for alert in self.active_alerts if not alert['acknowledged']]
        
        logger.info(f"🧹 Cleared alerts older than {days} days")

# ایجاد نمونه گلوبال
alert_manager = AlertManager()
