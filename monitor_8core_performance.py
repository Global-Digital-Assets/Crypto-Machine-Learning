#!/usr/bin/env python3
"""
8-Core Performance Monitor

Monitors CPU and memory usage of ML processes to ensure optimal 8-core utilization.
"""

import time
import psutil
import subprocess
import json
from datetime import datetime

def get_ml_processes():
    """Get all ML-related processes"""
    ml_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
            if any(keyword in cmdline.lower() for keyword in 
                   ['ml-', 'lightgbm', 'generate_daily_signals', 'continuous_learner', 
                    'analytics-api', 'data-api', 'futures-bot']):
                ml_processes.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'cpu_percent': proc.info['cpu_percent'],
                    'memory_percent': proc.info['memory_percent'],
                    'cmdline': cmdline[:100] + '...' if len(cmdline) > 100 else cmdline
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    return ml_processes

def monitor_performance():
    """Monitor system performance"""
    
    print("🔍 8-Core ML System Performance Monitor")
    print("=" * 60)
    
    # CPU info
    cpu_count = psutil.cpu_count()
    cpu_usage = psutil.cpu_percent(interval=1, percpu=True)
    
    print(f"Total CPUs: {cpu_count}")
    print(f"CPU Usage per core: {[f'{usage:.1f}%' for usage in cpu_usage]}")
    print(f"Average CPU Usage: {sum(cpu_usage)/len(cpu_usage):.1f}%")
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"Memory: {memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB ({memory.percent:.1f}%)")
    
    # ML processes
    ml_procs = get_ml_processes()
    if ml_procs:
        print("\nML Processes:")
        for proc in ml_procs:
            print(f"  PID {proc['pid']}: {proc['name']} - CPU: {proc['cpu_percent']:.1f}% - MEM: {proc['memory_percent']:.1f}%")
            print(f"    {proc['cmdline']}")
    
    print("=" * 60)

if __name__ == "__main__":
    try:
        while True:
            monitor_performance()
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")
