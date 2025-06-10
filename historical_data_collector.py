#!/usr/bin/env python3
"""
6-Month Historical Data Collector for Crypto ML Engine
Collects comprehensive historical data for all trading symbols
"""

import asyncio
import aiohttp
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from typing import List, Dict
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HistoricalDataCollector:
    def __init__(self, db_path: str = "/root/data-service/market_data.db"):
        self.db_path = db_path
        self.base_url = "https://fapi.binance.com"
        self.session = None
        
        # Read symbols from bucket mapping
        self.symbols = self._get_symbols_from_bucket_mapping()
        logger.info(f"Loaded {len(self.symbols)} symbols for historical collection")
        
        # 6 months back from today
        self.end_time = int(datetime.now().timestamp() * 1000)
        self.start_time = int((datetime.now() - timedelta(days=180)).timestamp() * 1000)
        
        logger.info(f"Collection period: {datetime.fromtimestamp(self.start_time/1000)} to {datetime.fromtimestamp(self.end_time/1000)}")
    
    def _get_symbols_from_bucket_mapping(self) -> List[str]:
        """Read symbols from bucket_mapping.csv"""
        try:
            # Try both local and remote paths
            local_path = "/Users/dom12/Desktop/Business/CRYPTO - MACHINE LEARNING/bucket_mapping.csv"
            remote_path = "/root/ml-engine/bucket_mapping.csv"
            
            csv_path = local_path if os.path.exists(local_path) else remote_path
            
            df = pd.read_csv(csv_path)
            symbols = df['symbol'].tolist()
            logger.info(f"Found symbols: {symbols}")
            return symbols
        except Exception as e:
            logger.error(f"Error reading bucket mapping: {e}")
            # Fallback to common crypto symbols
            return [
                'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT', 'BNBUSDT',
                'XRPUSDT', 'DOGEUSDT', 'MATICUSDT', 'AVAXUSDT', 'LINKUSDT'
            ]
    
    def _init_database(self):
        """Initialize database with proper schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create candles table (1-minute data)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS candles (
                symbol TEXT,
                timestamp INTEGER,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, timestamp)
            )
        ''')
        
        # Create 15-minute aggregated data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS candles_900s (
                symbol TEXT,
                timestamp INTEGER,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, timestamp)
            )
        ''')
        
        # Create futures metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS futures_microstructure (
                symbol TEXT,
                timestamp INTEGER,
                funding_rate REAL,
                funding_8h_sum REAL,
                funding_24h_sum REAL,
                funding_zscore REAL,
                oi_usd REAL,
                oi_1h_change_pct REAL,
                oi_5m_change_pct REAL,
                oi_zscore REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, timestamp)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database schema initialized")
    
    async def _fetch_candles(self, symbol: str, interval: str, limit: int = 1000, start_time: int = None, end_time: int = None) -> List[Dict]:
        """Fetch candlestick data from Binance"""
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        
        url = f"{self.base_url}/fapi/v1/klines"
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    candles = []
                    for kline in data:
                        candles.append({
                            'symbol': symbol,
                            'timestamp': int(kline[0]),
                            'open': float(kline[1]),
                            'high': float(kline[2]),
                            'low': float(kline[3]),
                            'close': float(kline[4]),
                            'volume': float(kline[5])
                        })
                    return candles
                else:
                    logger.error(f"Error fetching {symbol} {interval}: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Exception fetching {symbol} {interval}: {e}")
            return []
    
    def _save_candles(self, candles: List[Dict], table_name: str = 'candles'):
        """Save candles to database with upsert logic"""
        if not candles:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Use INSERT OR REPLACE to handle duplicates
        cursor.executemany(f'''
            INSERT OR REPLACE INTO {table_name} 
            (symbol, timestamp, open, high, low, close, volume) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', [
            (c['symbol'], c['timestamp'], c['open'], c['high'], c['low'], c['close'], c['volume'])
            for c in candles
        ])
        
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(candles)} {table_name} records for {candles[0]['symbol'] if candles else 'unknown'}")
    
    async def collect_historical_data(self, symbol: str):
        """Collect 6 months of historical data for a symbol"""
        logger.info(f"Starting historical collection for {symbol}")
        
        # Collect 1-minute data in chunks (max 1000 per request)
        current_start = self.start_time
        total_1m_candles = 0
        
        while current_start < self.end_time:
            # Calculate chunk end time (1000 minutes = ~16.7 hours)
            chunk_end = min(current_start + (1000 * 60 * 1000), self.end_time)
            
            # Fetch 1-minute data
            candles_1m = await self._fetch_candles(
                symbol, '1m', 1000, current_start, chunk_end
            )
            
            if candles_1m:
                self._save_candles(candles_1m, 'candles')
                total_1m_candles += len(candles_1m)
            
            # Rate limiting
            await asyncio.sleep(0.1)  # 100ms between requests
            current_start = chunk_end + 1
        
        # Collect 15-minute data in larger chunks
        current_start = self.start_time
        total_15m_candles = 0
        
        while current_start < self.end_time:
            # Calculate chunk end time (1000 * 15 minutes = 10.4 days)
            chunk_end = min(current_start + (1000 * 15 * 60 * 1000), self.end_time)
            
            # Fetch 15-minute data
            candles_15m = await self._fetch_candles(
                symbol, '15m', 1000, current_start, chunk_end
            )
            
            if candles_15m:
                self._save_candles(candles_15m, 'candles_900s')
                total_15m_candles += len(candles_15m)
            
            # Rate limiting
            await asyncio.sleep(0.1)
            current_start = chunk_end + 1
        
        logger.info(f"Completed {symbol}: {total_1m_candles} 1m candles, {total_15m_candles} 15m candles")
    
    async def run_collection(self):
        """Run historical data collection for all symbols"""
        logger.info("Starting 6-month historical data collection")
        
        # Initialize database
        self._init_database()
        
        # Create HTTP session
        connector = aiohttp.TCPConnector(limit=10)
        self.session = aiohttp.ClientSession(connector=connector)
        
        try:
            # Process symbols in small batches to avoid rate limits
            batch_size = 3
            for i in range(0, len(self.symbols), batch_size):
                batch = self.symbols[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}: {batch}")
                
                # Process batch concurrently
                tasks = [self.collect_historical_data(symbol) for symbol in batch]
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # Wait between batches
                if i + batch_size < len(self.symbols):
                    logger.info("Waiting 60 seconds between batches...")
                    await asyncio.sleep(60)
        
        finally:
            await self.session.close()
        
        logger.info("Historical data collection completed!")
        
        # Final verification
        self._verify_data_collection()
    
    def _verify_data_collection(self):
        """Verify the collected data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check 1-minute data
        cursor.execute('SELECT symbol, COUNT(*) FROM candles GROUP BY symbol')
        candles_1m = cursor.fetchall()
        
        # Check 15-minute data
        cursor.execute('SELECT symbol, COUNT(*) FROM candles_900s GROUP BY symbol')
        candles_15m = cursor.fetchall()
        
        logger.info("=== DATA COLLECTION SUMMARY ===")
        logger.info(f"1-minute data:")
        for symbol, count in candles_1m:
            expected_1m = 180 * 24 * 60  # 6 months * 24h * 60min = ~259,200
            coverage_pct = (count / expected_1m) * 100
            logger.info(f"  {symbol}: {count:,} candles ({coverage_pct:.1f}% of expected)")
        
        logger.info(f"15-minute data:")
        for symbol, count in candles_15m:
            expected_15m = 180 * 24 * 4  # 6 months * 24h * 4 (15min periods) = ~17,280
            coverage_pct = (count / expected_15m) * 100
            logger.info(f"  {symbol}: {count:,} candles ({coverage_pct:.1f}% of expected)")
        
        conn.close()

if __name__ == "__main__":
    collector = HistoricalDataCollector()
    asyncio.run(collector.run_collection())
