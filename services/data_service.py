"""
Data service for fetching historical market data using configured APIs
"""
import aiohttp
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from models.backtest import AssetData, PriceData
import logging

logger = logging.getLogger(__name__)

class DataService:
    """Service for fetching historical market data from multiple APIs"""
    
    def __init__(self):
        # API Keys (hardcoded as per configuration)
        self.fmp_key = "d853e9baf4d3b05e22df89d16fef4631"
        self.alpha_vantage_key = "W0TKFSHUKOR9TXN3"
        self.yahoo_key = "648b1a216amshf8a96de07812e45p14c7a6jsn4d44e897881f"
        self.fred_key = "26bbc1665befd935b8d8c55ae6e08ba8"
        
        # API endpoints
        self.fmp_base = "https://financialmodelingprep.com/api/v3"
        self.alpha_vantage_base = "https://www.alphavantage.co/query"
        self.yahoo_base = "https://yahoo-finance15.p.rapidapi.com"
        self.fred_base = "https://api.stlouisfed.org/fred"
    
    async def fetch_historical_data(
        self, 
        symbols: List[str], 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, AssetData]:
        """
        Fetch historical data for multiple symbols
        
        Args:
            symbols: List of asset symbols
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Dictionary mapping symbols to their historical data
        """
        results = {}
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for symbol in symbols:
                task = self._fetch_symbol_data(session, symbol, start_date, end_date)
                tasks.append(task)
            
            symbol_data = await asyncio.gather(*tasks, return_exceptions=True)
            
            for symbol, data in zip(symbols, symbol_data):
                if isinstance(data, Exception):
                    logger.error(f"Error fetching data for {symbol}: {data}")
                    # Create empty data as fallback
                    results[symbol] = AssetData(symbol=symbol, prices=[])
                else:
                    results[symbol] = data
        
        return results
    
    async def _fetch_symbol_data(
        self, 
        session: aiohttp.ClientSession, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> AssetData:
        """Fetch data for a single symbol using multiple API fallbacks"""
        
        # Try FMP first (most reliable for stocks)
        try:
            data = await self._fetch_from_fmp(session, symbol, start_date, end_date)
            if data and len(data.prices) > 0:
                return data
        except Exception as e:
            logger.warning(f"FMP failed for {symbol}: {e}")
        
        # Fallback to Alpha Vantage
        try:
            data = await self._fetch_from_alpha_vantage(session, symbol, start_date, end_date)
            if data and len(data.prices) > 0:
                return data
        except Exception as e:
            logger.warning(f"Alpha Vantage failed for {symbol}: {e}")
        
        # Fallback to Yahoo Finance
        try:
            data = await self._fetch_from_yahoo(session, symbol, start_date, end_date)
            if data and len(data.prices) > 0:
                return data
        except Exception as e:
            logger.warning(f"Yahoo Finance failed for {symbol}: {e}")
        
        # If all APIs fail, return empty data
        logger.error(f"All APIs failed for symbol {symbol}")
        return AssetData(symbol=symbol, prices=[])
    
    async def _fetch_from_fmp(
        self, 
        session: aiohttp.ClientSession, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> AssetData:
        """Fetch data from Financial Modeling Prep API"""
        
        url = f"{self.fmp_base}/historical-price-full/{symbol}"
        params = {
            "apikey": self.fmp_key,
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d")
        }
        
        async with session.get(url, params=params) as response:
            if response.status != 200:
                raise Exception(f"FMP API returned status {response.status}")
            
            data = await response.json()
            
            if "historical" not in data:
                raise Exception("No historical data in FMP response")
            
            prices = []
            for item in reversed(data["historical"]):  # FMP returns newest first
                try:
                    price = PriceData(
                        date=datetime.strptime(item["date"], "%Y-%m-%d"),
                        open=float(item["open"]),
                        high=float(item["high"]),
                        low=float(item["low"]),
                        close=float(item["close"]),
                        volume=int(item["volume"]),
                        adjusted_close=float(item.get("adjClose", item["close"]))
                    )
                    prices.append(price)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping invalid FMP data point for {symbol}: {e}")
                    continue
            
            return AssetData(symbol=symbol, prices=prices)
    
    async def _fetch_from_alpha_vantage(
        self, 
        session: aiohttp.ClientSession, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> AssetData:
        """Fetch data from Alpha Vantage API"""
        
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "apikey": self.alpha_vantage_key,
            "outputsize": "full"
        }
        
        async with session.get(self.alpha_vantage_base, params=params) as response:
            if response.status != 200:
                raise Exception(f"Alpha Vantage API returned status {response.status}")
            
            data = await response.json()
            
            if "Error Message" in data:
                raise Exception(f"Alpha Vantage error: {data['Error Message']}")
            
            if "Time Series (Daily)" not in data:
                raise Exception("No time series data in Alpha Vantage response")
            
            time_series = data["Time Series (Daily)"]
            prices = []
            
            for date_str, values in time_series.items():
                try:
                    date = datetime.strptime(date_str, "%Y-%m-%d")
                    
                    # Filter by date range
                    if date < start_date or date > end_date:
                        continue
                    
                    price = PriceData(
                        date=date,
                        open=float(values["1. open"]),
                        high=float(values["2. high"]),
                        low=float(values["3. low"]),
                        close=float(values["4. close"]),
                        volume=int(values["6. volume"]),
                        adjusted_close=float(values["5. adjusted close"])
                    )
                    prices.append(price)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping invalid Alpha Vantage data point for {symbol}: {e}")
                    continue
            
            # Sort by date
            prices.sort(key=lambda x: x.date)
            
            return AssetData(symbol=symbol, prices=prices)
    
    async def _fetch_from_yahoo(
        self, 
        session: aiohttp.ClientSession, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> AssetData:
        """Fetch data from Yahoo Finance API via RapidAPI"""
        
        # Convert dates to timestamps
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())
        
        url = f"{self.yahoo_base}/api/yahoo/hi/history/{symbol}/1d/{start_timestamp}/{end_timestamp}"
        
        headers = {
            "X-RapidAPI-Key": self.yahoo_key,
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
        }
        
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                raise Exception(f"Yahoo Finance API returned status {response.status}")
            
            data = await response.json()
            
            if "items" not in data:
                raise Exception("No items in Yahoo Finance response")
            
            prices = []
            for item in data["items"]:
                try:
                    price = PriceData(
                        date=datetime.fromtimestamp(item["date"]),
                        open=float(item["open"]),
                        high=float(item["high"]),
                        low=float(item["low"]),
                        close=float(item["close"]),
                        volume=int(item["volume"]),
                        adjusted_close=float(item.get("adjclose", item["close"]))
                    )
                    prices.append(price)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping invalid Yahoo data point for {symbol}: {e}")
                    continue
            
            # Sort by date
            prices.sort(key=lambda x: x.date)
            
            return AssetData(symbol=symbol, prices=prices)
    
    async def fetch_benchmark_data(
        self, 
        benchmark_symbol: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Optional[AssetData]:
        """Fetch benchmark data (e.g., SPY, QQQ)"""
        try:
            data = await self.fetch_historical_data([benchmark_symbol], start_date, end_date)
            return data.get(benchmark_symbol)
        except Exception as e:
            logger.error(f"Failed to fetch benchmark data for {benchmark_symbol}: {e}")
            return None
    
    def validate_data_quality(self, asset_data: AssetData, min_data_points: int = 50) -> Tuple[bool, List[str]]:
        """
        Validate the quality of fetched data
        
        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []
        
        if len(asset_data.prices) < min_data_points:
            warnings.append(f"Insufficient data points for {asset_data.symbol}: {len(asset_data.prices)} < {min_data_points}")
        
        # Check for missing data (gaps > 7 days)
        if len(asset_data.prices) > 1:
            for i in range(1, len(asset_data.prices)):
                gap = (asset_data.prices[i].date - asset_data.prices[i-1].date).days
                if gap > 7:  # More than a week gap
                    warnings.append(f"Data gap detected for {asset_data.symbol}: {gap} days between {asset_data.prices[i-1].date.date()} and {asset_data.prices[i].date.date()}")
        
        # Check for zero or negative prices
        invalid_prices = [p for p in asset_data.prices if p.close <= 0 or p.open <= 0]
        if invalid_prices:
            warnings.append(f"Invalid price data detected for {asset_data.symbol}: {len(invalid_prices)} data points")
        
        # Check for extreme price movements (> 50% in one day)
        extreme_moves = []
        for price in asset_data.prices:
            if price.high > 0 and price.low > 0:
                daily_range = (price.high - price.low) / price.low
                if daily_range > 0.5:  # 50% daily range
                    extreme_moves.append(price.date)
        
        if extreme_moves:
            warnings.append(f"Extreme price movements detected for {asset_data.symbol} on {len(extreme_moves)} days")
        
        is_valid = len(asset_data.prices) >= min_data_points and not any("Insufficient data" in w for w in warnings)
        
        return is_valid, warnings

# Global instance
data_service = DataService()

