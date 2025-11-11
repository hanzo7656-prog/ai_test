#!/usr/bin/env python3
"""
Auto Cleanup Script for VortexAI Project on Render.com
Run this periodically to free up disk space
"""

import os
import glob
import shutil
from datetime import datetime, timedelta
import psutil

class VortexCleanup:
    def __init__(self):
        self.deleted_count = 0
        self.freed_space = 0
        
    def log_action(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
        
    def get_file_size(self, filepath):
        try:
            return os.path.getsize(filepath)
        except:
            return 0
            
    def cleanup_pycache(self):
        """Clean Python cache directories"""
        print("🔍 Searching for pycache folders...")
        for pycache in glob.glob("**/__pycache__", recursive=True):
            try:
                if os.path.exists(pycache):
                    size_before = sum(self.get_file_size(os.path.join(pycache, f)) 
                                    for f in os.listdir(pycache) 
                                    if os.path.isfile(os.path.join(pycache, f)))
                    shutil.rmtree(pycache)
                    self.deleted_count += 1
                    self.freed_space += size_before
                    self.log_action(f"🧹 Removed pycache: {pycache} ({size_before/1024:.1f} KB)")
            except Exception as e:
                self.log_action(f"⚠️  Could not remove {pycache}: {e}")
                
    def cleanup_logs(self):
        """Clean log files older than 1 day"""
        print("🔍 Searching for log files...")
        log_patterns = [
            "*.log",
            "*.log.*",
            "debug_system/storage/*.log",
            "**/*.log"
        ]
        
        for pattern in log_patterns:
            for log_file in glob.glob(pattern, recursive=True):
                if os.path.isfile(log_file):
                    try:
                        # Delete logs older than 1 day
                        file_age = datetime.now() - datetime.fromtimestamp(
                            os.path.getmtime(log_file)
                        )
                        if file_age.days >= 1:
                            file_size = self.get_file_size(log_file)
                            os.remove(log_file)
                            self.deleted_count += 1
                            self.freed_space += file_size
                            self.log_action(f"🗑️  Removed old log: {log_file} ({file_size/1024:.1f} KB)")
                    except Exception as e:
                        self.log_action(f"⚠️  Could not remove {log_file}: {e}")
                        
    def cleanup_temp_files(self):
        """Clean temporary files"""
        print("🔍 Searching for temporary files...")
        temp_patterns = [
            "*.tmp",
            "*.temp",
            "temp/*",
            "cache/*",
            "**/*.pyc"
        ]
        
        for pattern in temp_patterns:
            for temp_file in glob.glob(pattern, recursive=True):
                if os.path.isfile(temp_file):
                    try:
                        file_size = self.get_file_size(temp_file)
                        os.remove(temp_file)
                        self.deleted_count += 1
                        self.freed_space += file_size
                        self.log_action(f"🔥 Removed temp file: {temp_file} ({file_size/1024:.1f} KB)")
                    except Exception as e:
                        self.log_action(f"⚠️  Could not remove {temp_file}: {e}")
                        
    def check_disk_usage(self):
        """Check current disk usage"""
        try:
            disk_usage = psutil.disk_usage('/')
            total_gb = disk_usage.total / (1024**3)
            used_gb = disk_usage.used / (1024**3)
            free_gb = disk_usage.free / (1024**3)
            
            self.log_action(f"💾 Disk Usage: {used_gb:.1f}GB / {total_gb:.1f}GB (Free: {free_gb:.1f}GB)")
            return free_gb
        except Exception as e:
            self.log_action(f"❌ Could not check disk usage: {e}")
            return 0
            
    def run_cleanup(self):
        """Run all cleanup tasks"""
        self.log_action("🚀 Starting VortexAI Cleanup...")
        
        free_before = self.check_disk_usage()
        
        self.cleanup_pycache()
        self.cleanup_logs() 
        self.cleanup_temp_files()
        
        free_after = self.check_disk_usage()
        
        self.log_action(f"🎉 Cleanup completed!")
        self.log_action(f"📊 Results: {self.deleted_count} files deleted, {self.freed_space/1024/1024:.2f} MB freed")
        
        if free_after > free_before:
            self.log_action(f"💫 Freed space: {(free_after - free_before):.2f} GB")
        else:
            self.log_action(f"💫 No significant space freed (maybe already clean)")
        
        return self.deleted_count

if __name__ == "__main__":
    cleaner = VortexCleanup()
    cleaner.run_cleanup()
