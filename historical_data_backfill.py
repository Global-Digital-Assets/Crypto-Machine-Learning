#!/usr/bin/env python3
"""
Historical Data Backfill Script for Institutional-Grade ML Training
Fetches 6+ months of 1-minute candle data from Binance API
Version: v1.0_institutional_backfill
"""

import asyncio
import aiohttp
import sqlite3
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backfill.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HistoricalDataBackfill:
    def __init__(self, db_path: str = 'market_data.db'):
        self.db_path = db_path
        self.session = None
        self.rate_limit_delay = 0.05  # 50ms between requests (1200/min max)
        
        # Target symbols for backfill
        self.symbols = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
            "ADAUSDT", "DOTUSDT", "AVAXUSDT", "LINKUSDT", "ATOMUSDT",
            "LTCUSDT", "ETCUSDT", "TRXUSDT", "NEARUSDT", "INJUSDT",
            "SUIUSDT", "AAVEUSDT", "SHIBUSDT", "DOGEUSDT", "PEPEUSDT",
            "OPUSDT", "APTUSDT", "ARBUSDT", "WLDUSDT", "FETUSDT",
            "SUSHIUSDT", "WIFUSDT", "TRUMPUSDT", "TAOUSDT", "TIAUSDT"
        ]
        
        self.stats = {
            'total_candles_fetched': 0,
            'total_requests': 0,
            'errors': 0,
            'symbols_completed': 0,
            'start_time': None
        }

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=10)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    def get_db_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)

    def get_existing_data_range(self, symbol: str) -> tuple:
        """Get existing data range for a symbol"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT MIN(timestamp), MAX(timestamp), COUNT(*)
                    FROM candles 
                    WHERE symbol = ?
                """, (symbol,))
                result = cursor.fetchone()
                
                if result and result[2] > 0:
                    return result[0], result[1], result[2]
                else:
                    return None, None, 0
        except Exception as e:
            logger.error(f"❌ Error checking existing data for {symbol}: {e}")
            return None, None, 0

    async def fetch_historical_candles(self, symbol: str, start_time: int, end_time: int, limit: int = 1000) -> List[Dict]:
        """Fetch historical candles from Binance API"""
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': '1m',
                'startTime': start_time,
                'endTime': end_time,
                'limit': limit
            }
            
            await asyncio.sleep(self.rate_limit_delay)  # Rate limiting
            
            async with self.session.get(url, params=params) as response:
                self.stats['total_requests'] += 1
                
                if response.status == 200:
                    data = await response.json()
                    candles = []
                    
                    for candle in data:
                        candles.append({
                            'symbol': symbol,
                            'timestamp': int(candle[0]),
                            'open': float(candle[1]),
                            'high': float(candle[2]),
                            'low': float(candle[3]),
                            'close': float(candle[4]),
                            'volume': float(candle[5])
                        })
                    
                    self.stats['total_candles_fetched'] += len(candles)
                    return candles
                
                elif response.status == 429:  # Rate limit hit
                    logger.warning(f"⚠️  Rate limit hit for {symbol}, waiting 60s...")
                    await asyncio.sleep(60)
                    return await self.fetch_historical_candles(symbol, start_time, end_time, limit)
                
                else:
                    logger.error(f"❌ API error for {symbol}: {response.status}")
                    self.stats['errors'] += 1
                    return []

        except Exception as e:
            logger.error(f"❌ Error fetching {symbol}: {e}")
            self.stats['errors'] += 1
            return []

    def insert_candles_batch(self, candles: List[Dict]) -> int:
        """Insert candles into database in batch"""
        if not candles:
            return 0
            
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Use INSERT OR IGNORE to handle duplicates
                cursor.executemany("""
                    INSERT OR IGNORE INTO candles 
                    (symbol, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, [
                    (c['symbol'], c['timestamp'], c['open'], c['high'], c['low'], c['close'], c['volume'])
                    for c in candles
                ])
                
                inserted = cursor.rowcount
                conn.commit()
                return inserted
                
        except Exception as e:
            logger.error(f"❌ Database error inserting batch: {e}")
            return 0

    async def backfill_symbol(self, symbol: str, months_back: int = 6) -> Dict:
        """Backfill historical data for a single symbol"""
        logger.info(f"🔄 Starting backfill for {symbol} ({months_back} months)")
        
        # Calculate time range
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=months_back * 30)).timestamp() * 1000)
        
        # Check existing data
        existing_min, existing_max, existing_count = self.get_existing_data_range(symbol)
        logger.info(f"📊 {symbol}: Existing data = {existing_count} candles")
        
        # Adjust start time if we have some data
        if existing_min:
            start_time = min(start_time, existing_min - 86400000)  # 1 day before existing data
            logger.info(f"📅 {symbol}: Adjusted start time to fill gaps")
        
        total_inserted = 0
        current_start = start_time
        batch_size = 1000  # Binance limit per request
        
        while current_start < end_time:
            # Calculate batch end time
            current_end = min(current_start + (batch_size * 60 * 1000), end_time)
            
            # Fetch batch
            candles = await self.fetch_historical_candles(symbol, current_start, current_end, batch_size)
            
            if candles:
                # Insert batch
                inserted = self.insert_candles_batch(candles)
                total_inserted += inserted
                
                # Log progress
                progress_pct = ((current_start - start_time) / (end_time - start_time)) * 100
                logger.info(f"📈 {symbol}: {progress_pct:.1f}% | Batch: {len(candles)} | Inserted: {inserted} | Total: {total_inserted}")
                
                # Update current start for next batch
                if candles:
                    current_start = candles[-1]['timestamp'] + 60000  # Next minute
                else:
                    current_start = current_end
            else:
                # No data returned, skip ahead
                current_start = current_end
                await asyncio.sleep(1)  # Brief pause on errors

        self.stats['symbols_completed'] += 1
        logger.info(f"✅ {symbol}: Completed! Inserted {total_inserted} new candles")
        
        return {
            'symbol': symbol,
            'inserted': total_inserted,
            'total_requests': self.stats['total_requests'],
            'errors': self.stats['errors']
        }

    async def run_backfill(self, months_back: int = 6, max_concurrent: int = 3):
        """Run complete backfill operation"""
        self.stats['start_time'] = datetime.now()
        logger.info(f"🚀 Starting historical data backfill for {len(self.symbols)} symbols")
        logger.info(f"📅 Target: {months_back} months of 1-minute candles")
        logger.info(f"🔄 Max concurrent: {max_concurrent} symbols")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_backfill(symbol):
            async with semaphore:
                return await self.backfill_symbol(symbol, months_back)
        
        # Run backfill for all symbols
        results = await asyncio.gather(
            *[limited_backfill(symbol) for symbol in self.symbols],
            return_exceptions=True
        )
        
        # Process results
        successful = 0
        total_inserted = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ {self.symbols[i]} failed: {result}")
            elif isinstance(result, dict):
                successful += 1
                total_inserted += result['inserted']
        
        # Final statistics
        duration = datetime.now() - self.stats['start_time']
        logger.info(f"🎉 Backfill completed!")
        logger.info(f"📊 Statistics:")
        logger.info(f"   ✅ Successful symbols: {successful}/{len(self.symbols)}")
        logger.info(f"   📈 Total candles inserted: {total_inserted:,}")
        logger.info(f"   🌐 Total API requests: {self.stats['total_requests']:,}")
        logger.info(f"   ❌ Errors: {self.stats['errors']}")
        logger.info(f"   ⏱️  Duration: {duration}")
        logger.info(f"   🚄 Rate: {total_inserted/duration.total_seconds():.1f} candles/second")
        
        return {
            'successful_symbols': successful,
            'total_inserted': total_inserted,
            'total_requests': self.stats['total_requests'],
            'errors': self.stats['errors'],
            'duration': duration
        }

    def validate_backfill(self) -> Dict:
        """Validate the backfilled data"""
        logger.info("🔍 Validating backfilled data...")
        
        validation_results = {}
        
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Overall statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_candles,
                        COUNT(DISTINCT symbol) as unique_symbols,
                        MIN(timestamp) as earliest_ms,
                        MAX(timestamp) as latest_ms,
                        datetime(MIN(timestamp)/1000, 'unixepoch') as earliest_date,
                        datetime(MAX(timestamp)/1000, 'unixepoch') as latest_date
                    FROM candles
                """)
                overall = cursor.fetchone()
                
                # Per-symbol statistics
                cursor.execute("""
                    SELECT 
                        symbol,
                        COUNT(*) as candle_count,
                        datetime(MIN(timestamp)/1000, 'unixepoch') as earliest,
                        datetime(MAX(timestamp)/1000, 'unixepoch') as latest,
                        ROUND((MAX(timestamp) - MIN(timestamp))/(1000*60*60*24.0), 1) as days_coverage
                    FROM candles 
                    GROUP BY symbol 
                    ORDER BY candle_count DESC
                """)
                per_symbol = cursor.fetchall()
                
                validation_results = {
                    'overall': {
                        'total_candles': overall[0],
                        'unique_symbols': overall[1],
                        'earliest_date': overall[4],
                        'latest_date': overall[5],
                        'date_range_days': (overall[3] - overall[2]) / (1000 * 60 * 60 * 24)
                    },
                    'per_symbol': [
                        {
                            'symbol': row[0],
                            'candle_count': row[1],
                            'earliest': row[2],
                            'latest': row[3],
                            'days_coverage': row[4]
                        }
                        for row in per_symbol
                    ]
                }
                
                # Log validation summary
                logger.info(f"📊 Validation Results:")
                logger.info(f"   📈 Total candles: {overall[0]:,}")
                logger.info(f"   🎯 Symbols: {overall[1]}")
                logger.info(f"   📅 Date range: {overall[4]} to {overall[5]}")
                logger.info(f"   ⏱️  Coverage: {validation_results['overall']['date_range_days']:.1f} days")
                
                # Check for symbols with insufficient data
                insufficient = [s for s in validation_results['per_symbol'] if s['candle_count'] < 10000]
                if insufficient:
                    logger.warning(f"⚠️  Symbols with < 10k candles: {[s['symbol'] for s in insufficient]}")
                
                return validation_results
                
        except Exception as e:
            logger.error(f"❌ Validation error: {e}")
            return {}

async def main():
    """Main execution function"""
    print("=" * 80)
    print("🏦 INSTITUTIONAL-GRADE HISTORICAL DATA BACKFILL")
    print("=" * 80)
    
    # Configuration
    months_back = 6  # 6 months of data
    max_concurrent = 3  # Gentle on Binance API
    db_path = 'market_data.db'
    
    try:
        async with HistoricalDataBackfill(db_path) as backfill:
            # Run backfill
            results = await backfill.run_backfill(months_back, max_concurrent)
            
            # Validate results
            validation = backfill.validate_backfill()
            
            # Save results
            with open('backfill_results.json', 'w') as f:
                json.dump({
                    'backfill_results': results,
                    'validation': validation,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2, default=str)
            
            logger.info("💾 Results saved to backfill_results.json")
            
            # Check if ready for ML training
            total_candles = validation.get('overall', {}).get('total_candles', 0)
            if total_candles > 500000:  # 500k+ candles for robust ML
                logger.info("🎉 Dataset ready for institutional-grade ML training!")
            else:
                logger.warning(f"⚠️  Dataset may be insufficient for robust ML ({total_candles:,} candles)")
                
    except Exception as e:
        logger.error(f"❌ Backfill failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
