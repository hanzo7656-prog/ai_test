import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import ipaddress

logger = logging.getLogger(__name__)

class SecurityMonitor:
    def __init__(self, alert_manager):
        self.alert_manager = alert_manager
        self.suspicious_activities = deque(maxlen=1000)
        self.failed_attempts = defaultdict(list)
        self.ip_whitelist = set()
        self.ip_blacklist = set()
        
        # الگوهای suspicious activity
        self.suspicious_patterns = {
            'sql_injection': [
                r"(\%27)|(\')|(\-\-)|(\%23)|(#)",
                r"((\%3D)|(=))[^\n]*((\%27)|(\')|(\-\-)|(\%3B)|(;))",
                r"\w*((\%27)|(\'))((\%6F)|o|(\%4F))((\%72)|r|(\%52))"
            ],
            'xss': [
                r"((\%3C)|<)((\%2F)|\/)*[a-z0-9\%]+((\%3E)|>)",
                r"((\%3C)|<)((\%69)|i|(\%49))((\%6D)|m|(\%4D))((\%67)|g|(\%47))[^\n]+((\%3E)|>)",
                r"((\%3C)|<)[^\n]+((\%3E)|>)"
            ],
            'path_traversal': [
                r"\.\.\/",
                r"\.\.\\",
                r"\/etc\/passwd",
                r"\/winnt\/win.ini"
            ]
        }
        
        self.rate_limits = {
            'general': {'limit': 100, 'window': 60},  # 100 requests per minute
            'auth': {'limit': 10, 'window': 60},      # 10 auth attempts per minute
            'api': {'limit': 1000, 'window': 300}     # 1000 API calls per 5 minutes
        }

        # اتصال به central_monitor برای دریافت alerts
        self._connect_to_central_monitor()
        
        logger.info("✅ Security Monitor Initialized - Central Monitor Connected")

    def _connect_to_central_monitor(self):
        """اتصال به central_monitor برای دریافت security alerts"""
        try:
            from .system_monitor import central_monitor
            
            if central_monitor:
                # عضویت برای دریافت security-related metrics
                central_monitor.subscribe("security_monitor", self._on_security_metrics_received)
                logger.info("✅ SecurityMonitor subscribed to central_monitor")
                
                # عضویت برای security alerts
                central_monitor.subscribe("security_monitor_alerts", self._on_security_alert_received)
                logger.info("✅ SecurityMonitor subscribed to security alerts")
            else:
                logger.warning("⚠️ Central monitor not available - security monitor will work independently")
                
        except ImportError:
            logger.warning("⚠️ Could not import central_monitor - security monitor will work independently")
        except Exception as e:
            logger.error(f"❌ Error connecting to central_monitor: {e}")

    def _on_security_metrics_received(self, metrics: Dict[str, Any]):
        """دریافت متریک‌های مرتبط با امنیت"""
        try:
            # می‌توانیم network metrics را برای تحلیل امنیتی استفاده کنیم
            network_metrics = metrics.get('system', {}).get('network', {})
            connections = network_metrics.get('connections', 0)
            
            # اگر connections غیرعادی زیاد باشد
            if connections > 1000:
                self._check_ddos_potential(connections, metrics)
                
        except Exception as e:
            logger.error(f"❌ Error processing security metrics: {e}")

    def _on_security_alert_received(self, alert_data: Dict[str, Any]):
        """دریافت security alerts از central_monitor"""
        try:
            # فقط لاگ کن
            logger.info(f"🛡️ Received security alert: {alert_data.get('title', 'No title')}")
            
            # اگر alert مربوط به IP blocking باشد
            if 'ip_address' in alert_data.get('data', {}):
                ip = alert_data['data']['ip_address']
                if alert_data.get('level') == 'CRITICAL':
                    self.add_to_blacklist(ip)
                    
        except Exception as e:
            logger.error(f"❌ Error processing security alert: {e}")

    def _check_ddos_potential(self, connections: int, metrics: Dict):
        """بررسی potential DDoS attack"""
        try:
            from debug_system.core.alert_manager import AlertLevel, AlertType
            
            if connections > 5000:
                self.alert_manager.create_alert(
                    level=AlertLevel.CRITICAL,
                    alert_type=AlertType.SECURITY,
                    title="Potential DDoS Attack Detected",
                    message=f"异常大量的连接数: {connections} - 可能遭受DDoS攻击",
                    source="security_monitor",
                    data={
                        'connections': connections,
                        'threshold': 5000,
                        'timestamp': datetime.now().isoformat()
                    }
                )
            elif connections > 2000:
                self.alert_manager.create_alert(
                    level=AlertLevel.WARNING,
                    alert_type=AlertType.SECURITY,
                    title="High Connection Count",
                    message=f"连接数异常高: {connections} - 请监控系统活动",
                    source="security_monitor",
                    data={
                        'connections': connections,
                        'threshold': 2000,
                        'timestamp': datetime.now().isoformat()
                    }
                )
                
        except Exception as e:
            logger.error(f"❌ Error checking DDoS potential: {e}")

    # بقیه متدها بدون تغییر (مثل قبل)
    def analyze_request(self, request_data: Dict) -> Dict[str, Any]:
        """آنالیز امنیتی درخواست"""
        security_analysis = {
            'threat_level': 'low',
            'warnings': [],
            'blocked': False,
            'timestamp': datetime.now().isoformat()
        }
        
        # بررسی IP
        ip_analysis = self._analyze_ip_address(request_data.get('client_ip'))
        if ip_analysis['threat_level'] != 'low':
            security_analysis['threat_level'] = ip_analysis['threat_level']
            security_analysis['warnings'].extend(ip_analysis['warnings'])
        
        # بررسی user agent
        ua_analysis = self._analyze_user_agent(request_data.get('user_agent'))
        if ua_analysis['threat_level'] != 'low':
            security_analysis['threat_level'] = max(
                security_analysis['threat_level'], 
                ua_analysis['threat_level']
            )
            security_analysis['warnings'].extend(ua_analysis['warnings'])
        
        # بررسی پارامترها
        param_analysis = self._analyze_parameters(request_data.get('params', {}))
        if param_analysis['threat_level'] != 'low':
            security_analysis['threat_level'] = max(
                security_analysis['threat_level'],
                param_analysis['threat_level']
            )
            security_analysis['warnings'].extend(param_analysis['warnings'])
        
        # بررسی rate limiting
        rate_analysis = self._check_rate_limits(request_data)
        if rate_analysis['threat_level'] != 'low':
            security_analysis['threat_level'] = max(
                security_analysis['threat_level'],
                rate_analysis['threat_level']
            )
            security_analysis['warnings'].extend(rate_analysis['warnings'])
        
        # تصمیم‌گیری نهایی
        if security_analysis['threat_level'] == 'high':
            security_analysis['blocked'] = True
            self._log_suspicious_activity(request_data, 'HIGH_THREAT_BLOCKED')
        elif security_analysis['threat_level'] == 'medium':
            self._log_suspicious_activity(request_data, 'MEDIUM_THREAT_DETECTED')
        
        return security_analysis

    def _analyze_ip_address(self, ip: str) -> Dict[str, Any]:
        """آنالیز آدرس IP"""
        if not ip:
            return {'threat_level': 'low', 'warnings': []}
        
        analysis = {'threat_level': 'low', 'warnings': []}
        
        try:
            ip_obj = ipaddress.ip_address(ip)
            
            # بررسی blacklist
            if ip in self.ip_blacklist:
                analysis['threat_level'] = 'high'
                analysis['warnings'].append('IP is in blacklist')
                return analysis
            
            # بررسی whitelist
            if ip in self.ip_whitelist:
                return analysis  # ایمن
            
            # بررسی IPهای خصوصی (ممکن است مشکوک باشند اگر از خارج انتظار می‌رود)
            if ip_obj.is_private:
                analysis['threat_level'] = 'medium'
                analysis['warnings'].append('Request from private IP address')
            
            # بررسی failed attempts
            recent_failures = [
                attempt for attempt in self.failed_attempts.get(ip, [])
                if datetime.now() - attempt < timedelta(hours=1)
            ]
            
            if len(recent_failures) > 10:
                analysis['threat_level'] = 'high'
                analysis['warnings'].append(f'Multiple recent failed attempts: {len(recent_failures)}')
            
        except ValueError:
            analysis['threat_level'] = 'high'
            analysis['warnings'].append('Invalid IP address format')
        
        return analysis

    def _analyze_user_agent(self, user_agent: str) -> Dict[str, Any]:
        """آنالیز User-Agent"""
        if not user_agent:
            return {'threat_level': 'low', 'warnings': []}
        
        analysis = {'threat_level': 'low', 'warnings': []}
        user_agent_lower = user_agent.lower()
        
        # الگوهای suspicious user agent
        suspicious_patterns = [
            'nmap', 'sqlmap', 'metasploit', 'nikto', 'wpscan', 
            'acunetix', 'appscan', 'burpsuite', 'zap'
        ]
        
        empty_or_missing = [
            '', 'unknown', 'undefined', 'none'
        ]
        
        # بررسی user agentهای خالی یا مشکوک
        if user_agent_lower in empty_or_missing:
            analysis['threat_level'] = 'medium'
            analysis['warnings'].append('Missing or generic User-Agent')
        
        # بررسی ابزارهای تست نفوذ
        for pattern in suspicious_patterns:
            if pattern in user_agent_lower:
                analysis['threat_level'] = 'high'
                analysis['warnings'].append(f'Suspicious User-Agent detected: {pattern}')
                break
        
        return analysis

    def _analyze_parameters(self, params: Dict) -> Dict[str, Any]:
        """آنالیز پارامترهای درخواست"""
        analysis = {'threat_level': 'low', 'warnings': []}
        
        for key, value in params.items():
            if not isinstance(value, str):
                continue
                
            param_analysis = self._check_injection_patterns(str(value))
            if param_analysis['threat_level'] != 'low':
                analysis['threat_level'] = max(
                    analysis['threat_level'],
                    param_analysis['threat_level']
                )
                analysis['warnings'].extend([
                    f"Parameter '{key}': {warning}" 
                    for warning in param_analysis['warnings']
                ])
        
        return analysis

    def _check_injection_patterns(self, value: str) -> Dict[str, Any]:
        """بررسی الگوهای injection"""
        analysis = {'threat_level': 'low', 'warnings': []}
        
        for attack_type, patterns in self.suspicious_patterns.items():
            for pattern in patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    analysis['threat_level'] = 'high'
                    analysis['warnings'].append(f'Potential {attack_type.upper()} detected')
                    break
        
        return analysis

    def _check_rate_limits(self, request_data: Dict) -> Dict[str, Any]:
        """بررسی rate limits"""
        analysis = {'threat_level': 'low', 'warnings': []}
        client_ip = request_data.get('client_ip')
        endpoint = request_data.get('endpoint')
        
        if not client_ip:
            return analysis
        
        current_time = datetime.now()
        
        # تعیین نوع rate limit
        limit_type = 'general'
        if endpoint and '/auth/' in endpoint:
            limit_type = 'auth'
        elif endpoint and '/api/' in endpoint:
            limit_type = 'api'
        
        limit_config = self.rate_limits[limit_type]
        window_start = current_time - timedelta(seconds=limit_config['window'])
        
        # شمارش درخواست‌ها در بازه زمانی
        request_count = sum(1 for activity in self.suspicious_activities
                          if activity.get('client_ip') == client_ip and
                          activity.get('timestamp', datetime.min) >= window_start)
        
        if request_count >= limit_config['limit']:
            analysis['threat_level'] = 'high'
            analysis['warnings'].append(
                f'Rate limit exceeded for {limit_type}: {request_count}/{limit_config["limit"]}'
            )
        
        return analysis

    def _log_suspicious_activity(self, request_data: Dict, activity_type: str):
        """ثبت فعالیت مشکوک"""
        activity = {
            'type': activity_type,
            'client_ip': request_data.get('client_ip'),
            'user_agent': request_data.get('user_agent'),
            'endpoint': request_data.get('endpoint'),
            'timestamp': datetime.now().isoformat(),
            'request_data': {
                k: v for k, v in request_data.items() 
                if k not in ['password', 'token', 'secret']
            }
        }
        
        self.suspicious_activities.append(activity)
        
        # ایجاد هشدار امنیتی
        self.alert_manager.create_alert(
            level='WARNING' if activity_type == 'MEDIUM_THREAT_DETECTED' else 'CRITICAL',
            alert_type='SECURITY',
            title=f"Security Alert: {activity_type}",
            message=f"Suspicious activity detected from {request_data.get('client_ip', 'unknown')}",
            source="security_monitor",
            data=activity
        )

    def log_failed_attempt(self, client_ip: str, reason: str):
        """ثبت تلاش ناموفق"""
        if not client_ip:
            return
            
        current_time = datetime.now()
        self.failed_attempts[client_ip].append(current_time)
        
        # پاکسازی تلاش‌های قدیمی
        cutoff_time = current_time - timedelta(hours=24)
        self.failed_attempts[client_ip] = [
            attempt for attempt in self.failed_attempts[client_ip]
            if attempt > cutoff_time
        ]
        
        # بررسی اگر تعداد تلاش‌ها زیاد باشد
        recent_attempts = [
            attempt for attempt in self.failed_attempts[client_ip]
            if attempt > current_time - timedelta(hours=1)
        ]
        
        if len(recent_attempts) > 5:
            self._log_suspicious_activity(
                {'client_ip': client_ip}, 
                'MULTIPLE_FAILED_ATTEMPTS'
            )

    def add_to_blacklist(self, ip: str):
        """اضافه کردن IP به blacklist"""
        self.ip_blacklist.add(ip)
        logger.warning(f"🚫 IP {ip} added to blacklist")

    def add_to_whitelist(self, ip: str):
        """اضافه کردن IP به whitelist"""
        self.ip_whitelist.add(ip)
        logger.info(f"✅ IP {ip} added to whitelist")

    def get_security_report(self, hours: int = 24) -> Dict[str, Any]:
        """دریافت گزارش امنیتی"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_activities = [
            activity for activity in self.suspicious_activities
            if datetime.fromisoformat(activity['timestamp']) >= cutoff_time
        ]
        
        # آمار بر اساس نوع فعالیت
        activity_stats = defaultdict(int)
        ip_stats = defaultdict(int)
        
        for activity in recent_activities:
            activity_stats[activity['type']] += 1
            ip_stats[activity['client_ip']] += 1
        
        # شناسایی IPهای پرخطر
        high_risk_ips = [
            ip for ip, count in ip_stats.items()
            if count >= 5  # بیش از ۵ فعالیت مشکوک
        ]
        
        return {
            'time_period_hours': hours,
            'total_suspicious_activities': len(recent_activities),
            'activity_breakdown': dict(activity_stats),
            'high_risk_ips': high_risk_ips,
            'blacklisted_ips_count': len(self.ip_blacklist),
            'whitelisted_ips_count': len(self.ip_whitelist),
            'recent_activities_sample': recent_activities[:10],  # نمونه‌ای از فعالیت‌ها
            'timestamp': datetime.now().isoformat()
        }

    def get_ip_reputation(self, ip: str) -> Dict[str, Any]:
        """دریافت اعتبار IP"""
        reputation = {
            'ip': ip,
            'risk_level': 'low',
            'factors': [],
            'statistics': {}
        }
        
        if not ip:
            return reputation
        
        # بررسی blacklist/whitelist
        if ip in self.ip_blacklist:
            reputation['risk_level'] = 'high'
            reputation['factors'].append('IP is blacklisted')
        
        if ip in self.ip_whitelist:
            reputation['risk_level'] = 'low'
            reputation['factors'].append('IP is whitelisted')
        
        # آمار فعالیت‌های مشکوک
        suspicious_count = sum(1 for activity in self.suspicious_activities
                              if activity.get('client_ip') == ip)
        reputation['statistics']['suspicious_activities'] = suspicious_count
        
        # آمار تلاش‌های ناموفق
        failed_count = len(self.failed_attempts.get(ip, []))
        reputation['statistics']['failed_attempts'] = failed_count
        
        # محاسبه نهایی سطح خطر
        if suspicious_count > 10 or failed_count > 20:
            reputation['risk_level'] = 'high'
        elif suspicious_count > 5 or failed_count > 10:
            reputation['risk_level'] = 'medium'
        
        reputation['factors'].extend([
            f'{suspicious_count} suspicious activities',
            f'{failed_count} failed attempts'
        ])
        
        return reputation

# ایجاد نمونه گلوبال (بعداً در main.py مقداردهی می‌شود)
security_monitor = None
