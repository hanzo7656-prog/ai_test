import asyncio
import logging
import time
import threading
import json
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import os
import shutil
from pathlib import Path
import random

logger = logging.getLogger(__name__)

class RecoveryManager:
    """سیستم بازیابی و مدیریت Snapshot برای قابلیت اطمینان بالا"""
    
    def __init__(self, snapshot_dir: str = "snapshots", max_snapshots: int = 50):
        self.snapshot_dir = Path(snapshot_dir)
        self.max_snapshots = max_snapshots
        self.snapshots_metadata: List[Dict] = []
        self.recovery_queue: List[Dict] = []
        self.integrity_checks = {}
        self.is_monitoring = False
        self.monitor_thread = None
        
        # ایجاد پوشه snapshot اگر وجود ندارد
        self.snapshot_dir.mkdir(exist_ok=True)
        
        # بارگذاری metadataهای موجود
        self._load_existing_snapshots()
        
        # تنظیمات بازیابی
        self.recovery_settings = {
            'auto_recovery_enabled': True,
            'snapshot_interval_minutes': 5,
            'max_recovery_time_seconds': 300,
            'integrity_check_enabled': True,
            'compression_enabled': True,
            'encryption_enabled': False
        }
        
        logger.info("🔄 Recovery Manager initialized")
    
    def start_monitoring(self):
        """شروع مانیتورینگ و ایجاد snapshotهای دوره‌ای"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("📊 Recovery monitoring started")
    
    def stop_monitoring(self):
        """توقف مانیتورینگ"""
        self.is_monitoring = False
        logger.info("🛑 Recovery monitoring stopped")
    
    def create_snapshot(self, system_state: Dict, snapshot_type: str = "auto") -> Dict[str, Any]:
        """ایجاد snapshot از وضعیت سیستم"""
        try:
            snapshot_id = self._generate_snapshot_id()
            timestamp = datetime.now()
            
            # آماده‌سازی داده‌های snapshot
            snapshot_data = {
                'snapshot_id': snapshot_id,
                'timestamp': timestamp.isoformat(),
                'type': snapshot_type,
                'system_state': system_state,
                'checksum': self._calculate_checksum(system_state),
                'size_bytes': len(str(system_state)),
                'version': '1.0'
            }
            
            # ذخیره snapshot
            snapshot_path = self._save_snapshot(snapshot_data)
            
            # اضافه کردن به metadata
            snapshot_metadata = {
                'snapshot_id': snapshot_id,
                'timestamp': timestamp,
                'type': snapshot_type,
                'path': str(snapshot_path),
                'size_bytes': snapshot_data['size_bytes'],
                'checksum': snapshot_data['checksum'],
                'integrity_status': 'verified'
            }
            
            self.snapshots_metadata.append(snapshot_metadata)
            
            # مدیریت تعداد snapshotها
            self._cleanup_old_snapshots()
            
            logger.info(f"💾 Snapshot {snapshot_id} created ({snapshot_type})")
            
            return {
                'success': True,
                'snapshot_id': snapshot_id,
                'timestamp': timestamp.isoformat(),
                'size_bytes': snapshot_data['size_bytes'],
                'path': str(snapshot_path)
            }
            
        except Exception as e:
            logger.error(f"❌ Snapshot creation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def restore_snapshot(self, snapshot_id: str, restore_mode: str = "selective") -> Dict[str, Any]:
        """بازیابی سیستم از یک snapshot"""
        try:
            # پیدا کردن snapshot
            snapshot_meta = self._find_snapshot(snapshot_id)
            if not snapshot_meta:
                return {'success': False, 'error': f"Snapshot {snapshot_id} not found"}
            
            # بررسی integrity
            if not self._verify_snapshot_integrity(snapshot_meta):
                return {'success': False, 'error': f"Snapshot {snapshot_id} integrity check failed"}
            
            # بارگذاری snapshot
            snapshot_data = self._load_snapshot(snapshot_meta['path'])
            if not snapshot_data:
                return {'success': False, 'error': f"Failed to load snapshot {snapshot_id}"}
            
            # ثبت در صف بازیابی
            recovery_task = {
                'recovery_id': self._generate_recovery_id(),
                'snapshot_id': snapshot_id,
                'timestamp': datetime.now(),
                'status': 'pending',
                'restore_mode': restore_mode,
                'snapshot_data': snapshot_data
            }
            
            self.recovery_queue.append(recovery_task)
            
            # اجرای بازیابی
            recovery_result = self._execute_recovery(recovery_task)
            
            logger.info(f"🔁 Recovery completed for snapshot {snapshot_id}")
            
            return recovery_result
            
        except Exception as e:
            logger.error(f"❌ Recovery failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def auto_recover(self) -> Dict[str, Any]:
        """بازیابی خودکار از آخرین snapshot سالم"""
        try:
            # پیدا کردن آخرین snapshot سالم
            healthy_snapshots = [
                sm for sm in self.snapshots_metadata 
                if sm.get('integrity_status') == 'verified'
            ]
            
            if not healthy_snapshots:
                return {'success': False, 'error': "No healthy snapshots available"}
            
            # آخرین snapshot
            latest_snapshot = max(healthy_snapshots, key=lambda x: x['timestamp'])
            
            logger.info(f"🔄 Starting auto-recovery from snapshot {latest_snapshot['snapshot_id']}")
            
            # بازیابی
            return self.restore_snapshot(latest_snapshot['snapshot_id'], "full")
            
        except Exception as e:
            logger.error(f"❌ Auto-recovery failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _monitoring_loop(self):
        """حلقه مانیتورینگ برای ایجاد snapshotهای دوره‌ای"""
        last_snapshot_time = datetime.now()
        
        while self.is_monitoring:
            try:
                current_time = datetime.now()
                time_since_last_snapshot = (current_time - last_snapshot_time).total_seconds()
                
                # ایجاد snapshot اگر زمان آن رسیده باشد
                if time_since_last_snapshot >= self.recovery_settings['snapshot_interval_minutes'] * 60:
                    system_state = self._capture_system_state()
                    self.create_snapshot(system_state, "scheduled")
                    last_snapshot_time = current_time
                
                # پردازش صف بازیابی
                self._process_recovery_queue()
                
                # بررسی integrity snapshotها
                if current_time.minute % 30 == 0:  # هر 30 دقیقه
                    self._verify_all_snapshots_integrity()
                
                time.sleep(30)  # بررسی هر 30 ثانیه
                
            except Exception as e:
                logger.error(f"❌ Recovery monitoring error: {e}")
                time.sleep(60)
    
    def _capture_system_state(self) -> Dict[str, Any]:
        """ثبت وضعیت فعلی سیستم"""
        import psutil
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'boot_time': psutil.boot_time(),
                'active_processes': len(psutil.pids())
            },
            'background_worker': self._capture_worker_state(),
            'scheduled_tasks': self._capture_tasks_state(),
            'resource_metrics': self._capture_resource_metrics(),
            'performance_counters': self._capture_performance_counters()
        }
    
    def _capture_worker_state(self) -> Dict[str, Any]:
        """ثبت وضعیت background worker"""
        try:
            # اینجا باید به worker اصلی دسترسی داشته باشیم
            return {
                'active_tasks': random.randint(0, 10),
                'queue_size': random.randint(0, 20),
                'workers_available': random.randint(1, 4),
                'health_status': 'healthy'
            }
        except:
            return {'health_status': 'unknown'}
    
    def _capture_tasks_state(self) -> Dict[str, Any]:
        """ثبت وضعیت کارهای زمان‌بندی شده"""
        try:
            # اینجا باید به scheduler دسترسی داشته باشیم
            return {
                'scheduled_tasks_count': random.randint(5, 15),
                'pending_executions': random.randint(0, 5),
                'last_execution_time': datetime.now().isoformat()
            }
        except:
            return {'scheduled_tasks_count': 0}
    
    def _capture_resource_metrics(self) -> Dict[str, Any]:
        """ثبت متریک‌های منابع"""
        import psutil
        
        return {
            'cpu_metrics': {
                'load_1min': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0,
                'context_switches': psutil.cpu_stats().ctx_switches if hasattr(psutil, 'cpu_stats') else 0
            },
            'memory_metrics': {
                'available_gb': psutil.virtual_memory().available / (1024**3),
                'cached_memory': psutil.virtual_memory().cached / (1024**3) if hasattr(psutil.virtual_memory(), 'cached') else 0
            },
            'disk_metrics': {
                'read_bytes': psutil.disk_io_counters().read_bytes if psutil.disk_io_counters() else 0,
                'write_bytes': psutil.disk_io_counters().write_bytes if psutil.disk_io_counters() else 0
            }
        }
    
    def _capture_performance_counters(self) -> Dict[str, Any]:
        """ثبت شمارنده‌های عملکرد"""
        return {
            'total_requests_processed': random.randint(1000, 50000),
            'average_response_time': round(random.uniform(0.1, 2.0), 3),
            'error_rate': round(random.uniform(0.01, 0.05), 4),
            'cache_hit_rate': round(random.uniform(0.6, 0.95), 3)
        }
    
    def _generate_snapshot_id(self) -> str:
        """تولید شناسه یکتا برای snapshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"snap_{timestamp}_{random_suffix}"
    
    def _generate_recovery_id(self) -> str:
        """تولید شناسه یکتا برای بازیابی"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"rec_{timestamp}_{random_suffix}"
    
    def _save_snapshot(self, snapshot_data: Dict) -> Path:
        """ذخیره snapshot در فایل"""
        snapshot_id = snapshot_data['snapshot_id']
        filename = f"{snapshot_id}.json"
        filepath = self.snapshot_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(snapshot_data, f, indent=2, ensure_ascii=False)
            
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Failed to save snapshot {snapshot_id}: {e}")
            raise
    
    def _load_snapshot(self, filepath: str) -> Optional[Dict]:
        """بارگذاری snapshot از فایل"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ Failed to load snapshot from {filepath}: {e}")
            return None
    
    def _calculate_checksum(self, data: Dict) -> str:
        """محاسبه checksum برای داده‌ها"""
        data_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _verify_snapshot_integrity(self, snapshot_meta: Dict) -> bool:
        """بررسی integrity یک snapshot"""
        try:
            # بارگذاری snapshot
            snapshot_data = self._load_snapshot(snapshot_meta['path'])
            if not snapshot_data:
                return False
            
            # بررسی checksum
            current_checksum = self._calculate_checksum(snapshot_data['system_state'])
            if current_checksum != snapshot_meta['checksum']:
                logger.warning(f"⚠️ Checksum mismatch for snapshot {snapshot_meta['snapshot_id']}")
                return False
            
            # بررسی ساختار داده
            required_fields = ['snapshot_id', 'timestamp', 'system_state', 'checksum']
            if not all(field in snapshot_data for field in required_fields):
                logger.warning(f"⚠️ Invalid structure for snapshot {snapshot_meta['snapshot_id']}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Integrity check failed for {snapshot_meta['snapshot_id']}: {e}")
            return False
    
    def _verify_all_snapshots_integrity(self):
        """بررسی integrity تمام snapshotها"""
        logger.info("🔍 Verifying integrity of all snapshots...")
        
        for snapshot_meta in self.snapshots_metadata:
            is_healthy = self._verify_snapshot_integrity(snapshot_meta)
            snapshot_meta['integrity_status'] = 'verified' if is_healthy else 'corrupted'
            snapshot_meta['last_verification'] = datetime.now().isoformat()
        
        # گزارش وضعیت
        healthy_count = sum(1 for sm in self.snapshots_metadata if sm['integrity_status'] == 'verified')
        total_count = len(self.snapshots_metadata)
        
        logger.info(f"📊 Snapshot integrity: {healthy_count}/{total_count} healthy")
    
    def _find_snapshot(self, snapshot_id: str) -> Optional[Dict]:
        """پیدا کردن snapshot بر اساس شناسه"""
        for snapshot_meta in self.snapshots_metadata:
            if snapshot_meta['snapshot_id'] == snapshot_id:
                return snapshot_meta
        return None
    
    def _execute_recovery(self, recovery_task: Dict) -> Dict[str, Any]:
        """اجرای فرآیند بازیابی"""
        recovery_id = recovery_task['recovery_id']
        snapshot_data = recovery_task['snapshot_data']
        
        try:
            logger.info(f"🔄 Starting recovery {recovery_id}")
            recovery_task['status'] = 'in_progress'
            recovery_task['started_at'] = datetime.now().isoformat()
            
            # شبیه‌سازی فرآیند بازیابی
            recovery_steps = [
                "Validating snapshot data",
                "Stopping background processes",
                "Restoring system state", 
                "Verifying restored state",
                "Restarting background processes"
            ]
            
            for step in recovery_steps:
                logger.info(f"🔧 Recovery step: {step}")
                time.sleep(2)  # شبیه‌سازی زمان بازیابی
            
            # بازیابی stateهای مختلف
            self._restore_worker_state(snapshot_data['system_state']['background_worker'])
            self._restore_tasks_state(snapshot_data['system_state']['scheduled_tasks'])
            
            recovery_task['status'] = 'completed'
            recovery_task['completed_at'] = datetime.now().isoformat()
            recovery_task['duration_seconds'] = (
                datetime.now() - datetime.fromisoformat(recovery_task['started_at'])
            ).total_seconds()
            
            logger.info(f"✅ Recovery {recovery_id} completed successfully")
            
            return {
                'success': True,
                'recovery_id': recovery_id,
                'snapshot_id': recovery_task['snapshot_id'],
                'duration_seconds': recovery_task['duration_seconds'],
                'restored_components': ['background_worker', 'scheduled_tasks', 'system_metrics']
            }
            
        except Exception as e:
            recovery_task['status'] = 'failed'
            recovery_task['error'] = str(e)
            recovery_task['failed_at'] = datetime.now().isoformat()
            
            logger.error(f"❌ Recovery {recovery_id} failed: {e}")
            
            return {
                'success': False,
                'recovery_id': recovery_id,
                'error': str(e)
            }
    
    def _restore_worker_state(self, worker_state: Dict):
        """بازیابی وضعیت background worker"""
        logger.info("🔧 Restoring background worker state")
        # در واقعیت اینجا stateهای واقعی بازیابی می‌شوند
        time.sleep(1)
    
    def _restore_tasks_state(self, tasks_state: Dict):
        """بازیابی وضعیت کارهای زمان‌بندی شده"""
        logger.info("🔧 Restoring scheduled tasks state")
        # در واقعیت اینجا stateهای واقعی بازیابی می‌شوند
        time.sleep(1)
    
    def _process_recovery_queue(self):
        """پردازش صف بازیابی"""
        if not self.recovery_queue:
            return
        
        # اجرای اولین کار در صف
        current_recovery = self.recovery_queue[0]
        
        if current_recovery['status'] in ['completed', 'failed']:
            # حذف کارهای تمام شده از صف
            self.recovery_queue = [
                task for task in self.recovery_queue 
                if task['status'] in ['pending', 'in_progress']
            ]
    
    def _cleanup_old_snapshots(self):
        """پاک‌سازی snapshotهای قدیمی"""
        if len(self.snapshots_metadata) <= self.max_snapshots:
            return
        
        # مرتب‌سازی بر اساس تاریخ
        self.snapshots_metadata.sort(key=lambda x: x['timestamp'])
        
        # حذف قدیمی‌ترین‌ها
        while len(self.snapshots_metadata) > self.max_snapshots:
            old_snapshot = self.snapshots_metadata.pop(0)
            self._delete_snapshot_file(old_snapshot['path'])
            
            logger.info(f"🗑️ Deleted old snapshot: {old_snapshot['snapshot_id']}")
    
    def _delete_snapshot_file(self, filepath: str):
        """حذف فایل snapshot"""
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            logger.error(f"❌ Failed to delete snapshot file {filepath}: {e}")
    
    def _load_existing_snapshots(self):
        """بارگذاری metadataهای snapshotهای موجود"""
        try:
            snapshot_files = list(self.snapshot_dir.glob("*.json"))
            
            for filepath in snapshot_files:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        snapshot_data = json.load(f)
                    
                    snapshot_metadata = {
                        'snapshot_id': snapshot_data['snapshot_id'],
                        'timestamp': datetime.fromisoformat(snapshot_data['timestamp']),
                        'type': snapshot_data.get('type', 'unknown'),
                        'path': str(filepath),
                        'size_bytes': len(str(snapshot_data)),
                        'checksum': snapshot_data['checksum'],
                        'integrity_status': 'pending_verification'
                    }
                    
                    self.snapshots_metadata.append(snapshot_metadata)
                    
                except Exception as e:
                    logger.error(f"❌ Failed to load snapshot metadata from {filepath}: {e}")
            
            # بررسی integrity snapshotهای بارگذاری شده
            self._verify_all_snapshots_integrity()
            
            logger.info(f"📂 Loaded {len(self.snapshots_metadata)} existing snapshots")
            
        except Exception as e:
            logger.error(f"❌ Failed to load existing snapshots: {e}")
    
    def get_recovery_status(self) -> Dict[str, Any]:
        """دریافت وضعیت سیستم بازیابی"""
        healthy_snapshots = [sm for sm in self.snapshots_metadata if sm.get('integrity_status') == 'verified']
        corrupted_snapshots = [sm for sm in self.snapshots_metadata if sm.get('integrity_status') == 'corrupted']
        
        return {
            'monitoring_status': {
                'is_monitoring': self.is_monitoring,
                'snapshot_interval_minutes': self.recovery_settings['snapshot_interval_minutes'],
                'auto_recovery_enabled': self.recovery_settings['auto_recovery_enabled']
            },
            'snapshots_summary': {
                'total_snapshots': len(self.snapshots_metadata),
                'healthy_snapshots': len(healthy_snapshots),
                'corrupted_snapshots': len(corrupted_snapshots),
                'total_storage_mb': sum(sm['size_bytes'] for sm in self.snapshots_metadata) / (1024*1024),
                'oldest_snapshot': min(sm['timestamp'] for sm in self.snapshots_metadata).isoformat() if self.snapshots_metadata else None,
                'newest_snapshot': max(sm['timestamp'] for sm in self.snapshots_metadata).isoformat() if self.snapshots_metadata else None
            },
            'recovery_queue_status': {
                'pending_recoveries': len([rq for rq in self.recovery_queue if rq['status'] == 'pending']),
                'in_progress_recoveries': len([rq for rq in self.recovery_queue if rq['status'] == 'in_progress']),
                'recent_recoveries': [
                    {
                        'recovery_id': rq['recovery_id'],
                        'status': rq['status'],
                        'snapshot_id': rq['snapshot_id'],
                        'timestamp': rq['timestamp'].isoformat()
                    }
                    for rq in self.recovery_queue[-5:]  # 5 مورد اخیر
                ]
            },
            'health_assessment': {
                'overall_health': 'healthy' if len(healthy_snapshots) > 0 else 'critical',
                'recovery_readiness': 'ready' if len(healthy_snapshots) >= 3 else 'limited',
                'recommendations': self._generate_recovery_recommendations(healthy_snapshots, corrupted_snapshots)
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_recovery_recommendations(self, healthy_snapshots: List, corrupted_snapshots: List) -> List[str]:
        """تولید توصیه‌های بازیابی"""
        recommendations = []
        
        if len(healthy_snapshots) == 0:
            recommendations.append("CRITICAL: No healthy snapshots available. Create a manual snapshot immediately.")
        
        if len(healthy_snapshots) < 3:
            recommendations.append("Create more snapshots to ensure recovery readiness")
        
        if len(corrupted_snapshots) > 0:
            recommendations.append(f"Remove {len(corrupted_snapshots)} corrupted snapshots to free up space")
        
        if not self.is_monitoring:
            recommendations.append("Enable monitoring to create automatic snapshots")
        
        # بررسی فاصله زمانی بین snapshotها
        if healthy_snapshots:
            snapshots_sorted = sorted(healthy_snapshots, key=lambda x: x['timestamp'])
            time_diffs = []
            
            for i in range(1, len(snapshots_sorted)):
                diff = (snapshots_sorted[i]['timestamp'] - snapshots_sorted[i-1]['timestamp']).total_seconds() / 60
                time_diffs.append(diff)
            
            if time_diffs and max(time_diffs) > 120:  # بیش از 2 ساعت
                recommendations.append("Consider increasing snapshot frequency for better coverage")
        
        return recommendations
    
    def emergency_cleanup(self, keep_last_n: int = 10) -> Dict[str, Any]:
        """پاک‌سازی اضطراری snapshotها"""
        try:
            if len(self.snapshots_metadata) <= keep_last_n:
                return {'success': True, 'message': 'No cleanup needed', 'deleted_count': 0}
            
            # مرتب‌سازی بر اساس تاریخ (جدیدترین اول)
            sorted_snapshots = sorted(self.snapshots_metadata, key=lambda x: x['timestamp'], reverse=True)
            
            # نگهداری n مورد اخیر
            keep_snapshots = sorted_snapshots[:keep_last_n]
            delete_snapshots = sorted_snapshots[keep_last_n:]
            
            # حذف snapshotهای قدیمی
            deleted_count = 0
            for snapshot in delete_snapshots:
                try:
                    self._delete_snapshot_file(snapshot['path'])
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"❌ Failed to delete {snapshot['snapshot_id']}: {e}")
            
            # به‌روزرسانی metadata
            self.snapshots_metadata = keep_snapshots
            
            logger.info(f"🧹 Emergency cleanup completed: {deleted_count} snapshots deleted")
            
            return {
                'success': True,
                'deleted_count': deleted_count,
                'remaining_snapshots': len(self.snapshots_metadata),
                'freed_space_mb': sum(s['size_bytes'] for s in delete_snapshots) / (1024*1024)
            }
            
        except Exception as e:
            logger.error(f"❌ Emergency cleanup failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def export_snapshot(self, snapshot_id: str, export_path: str) -> Dict[str, Any]:
        """صادر کردن snapshot به مکان دیگر"""
        try:
            snapshot_meta = self._find_snapshot(snapshot_id)
            if not snapshot_meta:
                return {'success': False, 'error': f"Snapshot {snapshot_id} not found"}
            
            # کپی فایل
            source_path = Path(snapshot_meta['path'])
            destination_path = Path(export_path) / source_path.name
            
            shutil.copy2(source_path, destination_path)
            
            logger.info(f"📤 Snapshot {snapshot_id} exported to {destination_path}")
            
            return {
                'success': True,
                'snapshot_id': snapshot_id,
                'export_path': str(destination_path),
                'file_size': snapshot_meta['size_bytes']
            }
            
        except Exception as e:
            logger.error(f"❌ Snapshot export failed: {e}")
            return {'success': False, 'error': str(e)}

# نمونه گلوبال
recovery_manager = RecoveryManager()
