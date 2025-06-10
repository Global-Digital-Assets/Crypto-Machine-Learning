#!/usr/bin/env python3
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
