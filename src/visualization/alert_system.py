# 📁 src/visualization/alert_system.py

import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import logging
from typing import Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)

class AlertSystem:
    """سیستم ارسال هشدار و نوتیفیکیشن"""
    
    def __init__(self, smtp_config: Dict = None):
        self.smtp_config = smtp_config or {}
        self.alert_history: List[Dict] = []
    
    def send_signal_alert(self, signal: Dict, recipients: List[str]):
        """ارسال هشدار سیگنال"""
        try:
            subject = f"🚨 Trading Signal: {signal['symbol']} - {signal['signal_type']}"
            
            message = f"""
            Trading Signal Alert
            
            Symbol: {signal['symbol']}
            Signal: {signal['signal_type']}
            Confidence: {signal['confidence']:.1%}
            Current Price: ${signal['price']:.2f}
            Targets: {[f'${t:.2f}' for t in signal['targets']]}
            Stop Loss: ${signal['stop_loss']:.2f}
            Risk/Reward: 1:{signal['risk_reward_ratio']:.1f}
            
            Reasons:
            {chr(10).join(f'• {reason}' for reason in signal['reasons'])}
            
            Time: {signal['timestamp']}
            """
            
            # ارسال ایمیل
            if self.smtp_config:
                self._send_email(subject, message, recipients)
            
            # ذخیره در تاریخچه
            self.alert_history.append({
                'type': 'signal',
                'symbol': signal['symbol'],
                'signal_type': signal['signal_type'],
                'confidence': signal['confidence'],
                'timestamp': datetime.now(),
                'message': message
            })
            
            logger.info(f"✅ Signal alert sent for {signal['symbol']}")
            
        except Exception as e:
            logger.error(f"Failed to send signal alert: {e}")
    
    def send_system_alert(self, alert_type: str, message: str, recipients: List[str]):
        """ارسال هشدار سیستم"""
        try:
            subject = f"⚠️ System Alert: {alert_type}"
            
            full_message = f"""
            System Alert
            
            Type: {alert_type}
            Message: {message}
            Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            Please check the system status.
            """
            
            if self.smtp_config:
                self._send_email(subject, full_message, recipients)
            
            self.alert_history.append({
                'type': 'system',
                'alert_type': alert_type,
                'message': message,
                'timestamp': datetime.now()
            })
            
            logger.info(f"✅ System alert sent: {alert_type}")
            
        except Exception as e:
            logger.error(f"Failed to send system alert: {e}")
    
    def send_performance_alert(self, metrics: Dict, recipients: List[str]):
        """ارسال هشدار عملکرد"""
        try:
            subject = "📊 Daily Performance Report"
            
            message = f"""
            Daily Performance Report
            
            Total Return: {metrics.get('total_return', 0):.2f}%
            Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
            Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%
            Win Rate: {metrics.get('win_rate', 0):.1f}%
            
            Total Trades: {metrics.get('total_trades', 0)}
            Winning Trades: {metrics.get('winning_trades', 0)}
            Losing Trades: {metrics.get('losing_trades', 0)}
            
            Report Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            if self.smtp_config:
                self._send_email(subject, message, recipients)
            
            logger.info("✅ Performance report sent")
            
        except Exception as e:
            logger.error(f"Failed to send performance alert: {e}")
    
    def _send_email(self, subject: str, body: str, recipients: List[str]):
        """ارسال ایمیل"""
        if not self.smtp_config:
            return
        
        try:
            msg = MimeMultipart()
            msg['From'] = self.smtp_config['from_email']
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            
            msg.attach(MimeText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_config['smtp_server'], self.smtp_config['smtp_port']) as server:
                server.starttls()
                server.login(self.smtp_config['username'], self.smtp_config['password'])
                server.send_message(msg)
                
        except Exception as e:
            logger.error(f"Email sending failed: {e}")
    
    def get_alert_history(self, hours: int = 24) -> List[Dict]:
        """دریافت تاریخچه هشدارها"""
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        return [
            alert for alert in self.alert_history
            if alert['timestamp'].timestamp() > cutoff_time
        ]
