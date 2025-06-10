#!/usr/bin/env python3
"""
8-Core Optimization Script for Crypto ML System

Optimizes LightGBM training and feature engineering to use all 8 cores
of the new VPS. Updates production scripts for maximum performance.
"""

import os
import sys
import shutil
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_lightgbm_params():
    """Update LightGBM parameters in production scripts for 8-core usage"""
    
    scripts_to_update = [
        'production_ml_pipeline.py',
        'generate_daily_signals.py', 
        'continuous_learner.py',
        'weekly_retrain_pipeline.py'
    ]
    
    for script in scripts_to_update:
        if Path(script).exists():
            logger.info(f"📝 Updating {script} for 8-core LightGBM...")
            
            # Read file
            with open(script, 'r') as f:
                content = f.read()
            
            # Update num_threads from 4 to 8
            content = content.replace("'num_threads': 4", "'num_threads': 8")
            content = content.replace('"num_threads": 4', '"num_threads": 8')
            
            # Add OpenMP environment if needed
            if 'os.environ' not in content and 'lightgbm' in content:
                import_section = content.find('import lightgbm')
                if import_section != -1:
                    insert_pos = content.find('\n', import_section) + 1
                    env_setup = '''
# Optimize for 8-core usage
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('MKL_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')

'''
                    content = content[:insert_pos] + env_setup + content[insert_pos:]
            
            # Write back
            with open(script, 'w') as f:
                f.write(content)
                
            logger.info(f"✅ Updated {script}")
        else:
            logger.warning(f"⚠️  {script} not found")

def create_parallel_training_script():
    """Create a parallel training script for multiple models"""
    
    script_content = '''#!/usr/bin/env python3
"""
Parallel ML Training Script - 8 Core Optimization

Trains multiple models in parallel using all 8 cores for maximum efficiency.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import lightgbm as lgb
import multiprocessing as mp

# Optimize for 8-core usage
os.environ.setdefault('OMP_NUM_THREADS', '2')  # 2 threads per worker
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

def train_symbol_model(symbol_data):
    """Train model for a single symbol using 2 cores"""
    symbol, X, y = symbol_data
    
    # LightGBM params optimized for parallel execution
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 100,
        'num_threads': 2,  # 2 threads per worker
        'verbosity': -1,
        'random_state': 42
    }
    
    train_data = lgb.Dataset(X, label=y)
    model = lgb.train(params, train_data, num_boost_round=200)
    
    return symbol, model

def parallel_model_training(symbol_datasets, max_workers=4):
    """Train multiple models in parallel"""
    
    results = {}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all training jobs
        future_to_symbol = {
            executor.submit(train_symbol_model, data): data[0] 
            for data in symbol_datasets
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                symbol_name, model = future.result()
                results[symbol_name] = model
                print(f"✅ Completed training for {symbol_name}")
            except Exception as exc:
                print(f"❌ {symbol} generated an exception: {exc}")
    
    return results

if __name__ == "__main__":
    print("🚀 8-Core Parallel ML Training")
    print(f"Available cores: {mp.cpu_count()}")
    print("This script trains 4 models simultaneously using 2 cores each")
'''
    
    with open('parallel_training_8core.py', 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod('parallel_training_8core.py', 0o755)
    logger.info("✅ Created parallel_training_8core.py")

def create_performance_monitor():
    """Create a performance monitoring script"""
    
    monitor_content = '''#!/usr/bin/env python3
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
        print("\\nML Processes:")
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
        print("\\n👋 Monitoring stopped")
'''
    
    with open('monitor_8core_performance.py', 'w') as f:
        f.write(monitor_content)
    
    os.chmod('monitor_8core_performance.py', 0o755)
    logger.info("✅ Created monitor_8core_performance.py")

def main():
    """Main optimization function"""
    
    logger.info("🚀 Starting 8-Core Optimization for Crypto ML System")
    
    # Update existing scripts
    update_lightgbm_params()
    
    # Create new parallel training script
    create_parallel_training_script()
    
    # Create performance monitor
    create_performance_monitor()
    
    logger.info("✅ 8-Core optimization complete!")
    logger.info("\nNext steps:")
    logger.info("1. Deploy updated scripts to VPS")
    logger.info("2. Restart ML services to use new parameters")
    logger.info("3. Run monitor_8core_performance.py to verify utilization")
    logger.info("4. Use parallel_training_8core.py for batch model training")

if __name__ == "__main__":
    main()
