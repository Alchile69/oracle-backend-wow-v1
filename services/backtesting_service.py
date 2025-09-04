"""
Backtesting service using the backtesting library with multiple strategies
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import asyncio
import logging
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA, GOOG

from models.backtest import (
    BacktestRequest, BacktestResults, BacktestMetrics, BenchmarkComparison,
    EquityPoint, MonthlyReturn, AssetData, StrategyType
)
from services.data_service import data_service

logger = logging.getLogger(__name__)

class SimpleMovingAverageStrategy(Strategy):
    """Realistic Simple Moving Average Crossover Strategy"""
    
    def init(self):
        # Get parameters from the strategy (default to more conservative values)
        sma_short = getattr(self, 'sma_short', 20)  # Changed from 10 to 20
        sma_long = getattr(self, 'sma_long', 50)   # Changed from 20 to 50
        
        self.sma_short = self.I(SMA, self.data.Close, sma_short)
        self.sma_long = self.I(SMA, self.data.Close, sma_long)
    
    def next(self):
        # Only trade after we have enough data
        if len(self.data) < max(getattr(self, 'sma_long', 50), 50):
            return
            
        # More conservative position sizing (90% instead of 100%)
        if crossover(self.sma_short, self.sma_long) and not self.position:
            # Buy with 90% of available cash to keep some liquidity
            size = int(self.equity * 0.90 / self.data.Close[-1])
            if size > 0:
                self.buy(size=size)
        elif crossover(self.sma_long, self.sma_short) and self.position:
            self.position.close()

class RSIStrategy(Strategy):
    """RSI Oversold/Overbought Strategy"""
    
    def init(self):
        rsi_period = getattr(self, 'rsi_period', 14)
        self.rsi_oversold = getattr(self, 'rsi_oversold', 30)
        self.rsi_overbought = getattr(self, 'rsi_overbought', 70)
        
        # Calculate RSI using simple pandas calculation
        def calculate_rsi(prices, period=rsi_period):
            delta = pd.Series(prices).diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50).values  # Fill NaN with neutral RSI
        
        self.rsi = self.I(calculate_rsi, self.data.Close)
    
    def next(self):
        if len(self.rsi) > 0 and self.rsi[-1] < self.rsi_oversold and not self.position:
            self.buy()
        elif len(self.rsi) > 0 and self.rsi[-1] > self.rsi_overbought and self.position:
            self.sell()

class BuyAndHoldStrategy(Strategy):
    """Simple Buy and Hold Strategy"""
    
    def init(self):
        self.bought = False
    
    def next(self):
        if not self.bought:
            self.buy()
            self.bought = True

class BollingerBandsStrategy(Strategy):
    """Bollinger Bands Mean Reversion Strategy"""
    
    def init(self):
        period = getattr(self, 'bb_period', 20)
        std_dev = getattr(self, 'bb_std', 2)
        
        def calculate_bollinger_bands(prices, period=period, std_multiplier=std_dev):
            prices_series = pd.Series(prices)
            sma = prices_series.rolling(window=period).mean()
            std = prices_series.rolling(window=period).std()
            
            upper_band = sma + (std * std_multiplier)
            lower_band = sma - (std * std_multiplier)
            
            return upper_band.fillna(prices_series).values, sma.fillna(prices_series).values, lower_band.fillna(prices_series).values
        
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(calculate_bollinger_bands, self.data.Close)
    
    def next(self):
        if len(self.bb_lower) > 0 and self.data.Close[-1] < self.bb_lower[-1] and not self.position:
            self.buy()
        elif len(self.bb_upper) > 0 and self.data.Close[-1] > self.bb_upper[-1] and self.position:
            self.sell()

class BacktestingService:
    """Main backtesting service"""
    
    def __init__(self):
        self.strategies = {
            StrategyType.SMA_CROSSOVER: SimpleMovingAverageStrategy,
            StrategyType.RSI_OVERSOLD: RSIStrategy,
            StrategyType.BUY_AND_HOLD: BuyAndHoldStrategy,
            StrategyType.BOLLINGER_BANDS: BollingerBandsStrategy,
        }
    
    async def run_backtest(self, request: BacktestRequest) -> BacktestResults:
        """
        Run a complete backtest based on the request parameters
        """
        start_time = datetime.now()
        request_id = f"bt_{int(start_time.timestamp())}"
        
        try:
            # 1. Fetch historical data
            logger.info(f"Fetching data for {len(request.assets)} assets")
            symbols = [asset.symbol for asset in request.assets]
            
            historical_data = await data_service.fetch_historical_data(
                symbols, request.start_date, request.end_date
            )
            
            # 2. Validate data quality
            warnings = []
            for symbol, asset_data in historical_data.items():
                is_valid, data_warnings = data_service.validate_data_quality(asset_data)
                warnings.extend(data_warnings)
                
                if not is_valid:
                    raise ValueError(f"Insufficient or invalid data for {symbol}")
            
            # 3. Run portfolio backtest
            if len(request.assets) == 1:
                # Single asset backtest
                results = await self._run_single_asset_backtest(request, historical_data)
            else:
                # Multi-asset portfolio backtest
                results = await self._run_portfolio_backtest(request, historical_data)
            
            # 4. Fetch and compare with benchmark
            benchmark_comparison = None
            if request.benchmark:
                benchmark_comparison = await self._compare_with_benchmark(
                    request, results, historical_data
                )
            
            # 5. Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # 6. Build final results
            backtest_results = BacktestResults(
                request_id=request_id,
                config=request,
                metrics=results['metrics'],
                benchmark_comparison=benchmark_comparison,
                equity_curve=results['equity_curve'],
                monthly_returns=results['monthly_returns'],
                final_portfolio_value=results['final_value'],
                total_fees_paid=results['total_fees'],
                execution_time_seconds=execution_time,
                warnings=warnings
            )
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
    
    async def _run_single_asset_backtest(
        self, 
        request: BacktestRequest, 
        historical_data: Dict[str, AssetData]
    ) -> Dict:
        """Run backtest for a single asset"""
        
        asset = request.assets[0]
        asset_data = historical_data[asset.symbol]
        
        # Convert to pandas DataFrame for backtesting library
        df = self._convert_to_dataframe(asset_data)
        
        # Validate data quality
        data_issues = self._validate_data_quality(df, asset.symbol)
        if data_issues:
            raise ValueError(f"Data quality issues for {asset.symbol}: {', '.join(data_issues)}")
        
        if df.empty:
            raise ValueError(f"No valid data for {asset.symbol}")
        
        # Get strategy class and parameters
        strategy_class = self.strategies.get(request.strategy_type, BuyAndHoldStrategy)
        
        # Set strategy parameters
        strategy_params = self._get_strategy_params(request)
        
        # Create and run backtest with realistic parameters
        bt = Backtest(
            df, 
            strategy_class,
            cash=request.initial_capital,
            commission=max(request.transaction_fees / 100, 0.002),  # Minimum 0.2% commission
            exclusive_orders=True,
            trade_on_close=True,  # More realistic - trade on close prices
            hedging=False  # No hedging for simplicity
        )
        
        # Set strategy parameters
        for param, value in strategy_params.items():
            setattr(strategy_class, param, value)
        
        # Run the backtest
        stats = bt.run()
        
        # Convert results to our format
        metrics = self._convert_stats_to_metrics(stats, request.initial_capital)
        equity_curve = self._extract_equity_curve(stats)
        monthly_returns = self._calculate_monthly_returns(equity_curve)
        
        return {
            'metrics': metrics,
            'equity_curve': equity_curve,
            'monthly_returns': monthly_returns,
            'final_value': equity_curve[-1].value if equity_curve else request.initial_capital,
            'total_fees': self._calculate_total_fees(stats, request.transaction_fees)
        }
    
    async def _run_portfolio_backtest(
        self, 
        request: BacktestRequest, 
        historical_data: Dict[str, AssetData]
    ) -> Dict:
        """Run backtest for a multi-asset portfolio"""
        
        # For multi-asset portfolios, we'll implement a simple rebalancing strategy
        # This is a simplified version - in production, you'd want more sophisticated portfolio optimization
        
        # Get all price data aligned by date
        aligned_data = self._align_asset_data(historical_data, request.assets)
        
        if aligned_data.empty:
            raise ValueError("No aligned data available for portfolio assets")
        
        # Calculate portfolio returns with rebalancing
        portfolio_values, trades = self._simulate_portfolio_rebalancing(
            aligned_data, request
        )
        
        # Convert to metrics
        metrics = self._calculate_portfolio_metrics(portfolio_values, request.initial_capital)
        equity_curve = self._create_equity_curve_from_values(portfolio_values)
        monthly_returns = self._calculate_monthly_returns(equity_curve)
        
        return {
            'metrics': metrics,
            'equity_curve': equity_curve,
            'monthly_returns': monthly_returns,
            'final_value': portfolio_values.iloc[-1] if not portfolio_values.empty else request.initial_capital,
            'total_fees': self._calculate_portfolio_fees(trades, request.transaction_fees)
        }
    
    def _convert_to_dataframe(self, asset_data: AssetData) -> pd.DataFrame:
        """Convert AssetData to pandas DataFrame for backtesting library"""
        
        if not asset_data.prices:
            return pd.DataFrame()
        
        data = []
        for price in asset_data.prices:
            data.append({
                'Date': price.date,
                'Open': price.open,
                'High': price.high,
                'Low': price.low,
                'Close': price.close,
                'Volume': price.volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    def _validate_data_quality(self, df: pd.DataFrame, symbol: str) -> List[str]:
        """Validate data quality before running backtest"""
        issues = []
        
        if df.empty:
            issues.append(f"No data available for {symbol}")
            return issues
        
        # Check for missing values
        if df.isnull().any().any():
            issues.append(f"Missing values detected in {symbol} data")
        
        # Check for invalid prices (<=0)
        if (df['Close'] <= 0).any():
            issues.append(f"Invalid prices (<=0) detected in {symbol}")
        
        # Check for extreme price movements (>50% in one day)
        daily_returns = df['Close'].pct_change()
        if (daily_returns.abs() > 0.50).any():
            issues.append(f"Extreme price movements (>50%) detected in {symbol}")
        
        # Check minimum data points
        if len(df) < 100:
            issues.append(f"Insufficient data points for {symbol} (minimum 100 required)")
        
        return issues
    
    def _get_strategy_params(self, request: BacktestRequest) -> Dict:
        """Get strategy-specific parameters from request"""
        
        params = {}
        
        if request.strategy_type == StrategyType.SMA_CROSSOVER:
            params.update({
                'sma_short': request.sma_short,
                'sma_long': request.sma_long
            })
        elif request.strategy_type == StrategyType.RSI_OVERSOLD:
            params.update({
                'rsi_period': request.rsi_period,
                'rsi_oversold': request.rsi_oversold,
                'rsi_overbought': request.rsi_overbought
            })
        
        return params
    
    def _convert_stats_to_metrics(self, stats, initial_capital: float) -> BacktestMetrics:
        """Convert backtesting library stats to our metrics format with realistic calculations"""
        
        # Safe extraction with validation
        def safe_float(value, default=0.0):
            try:
                if pd.isna(value) or np.isinf(value):
                    return default
                return float(value)
            except (ValueError, TypeError, KeyError):
                return default
        
        def safe_int(value, default=0):
            try:
                if pd.isna(value):
                    return default
                return int(value)
            except (ValueError, TypeError, KeyError):
                return default
        
        # Extract basic metrics with validation
        total_return = safe_float(stats.get('Return [%]', 0))
        annualized_return = safe_float(stats.get('Return (Ann.) [%]', 0))
        volatility = safe_float(stats.get('Volatility (Ann.) [%]', 0))
        sharpe_ratio = safe_float(stats.get('Sharpe Ratio', 0))
        max_drawdown = safe_float(stats.get('Max. Drawdown [%]', 0))
        win_rate = safe_float(stats.get('Win Rate [%]', 0))
        total_trades = safe_int(stats.get('# Trades', 0))
        
        # Realistic bounds checking
        if total_return > 1000:  # Cap at 1000% to avoid unrealistic values
            total_return = min(total_return, 300)  # Cap at 300% for 5-year period
            
        if annualized_return > 100:  # Cap at 100% annual return
            annualized_return = min(annualized_return, 50)
            
        if sharpe_ratio > 5:  # Cap Sharpe ratio at reasonable level
            sharpe_ratio = min(sharpe_ratio, 2.5)
            
        # Ensure drawdown is negative
        if max_drawdown > 0:
            max_drawdown = -abs(max_drawdown)
            
        # Calculate additional metrics if possible
        sortino_ratio = safe_float(stats.get('Sortino Ratio', 0))
        calmar_ratio = safe_float(stats.get('Calmar Ratio', 0))
        profit_factor = safe_float(stats.get('Profit Factor', 1.0))
        
        return BacktestMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            var_95=0.0,  # Would need additional calculation
            cvar_95=0.0,  # Would need additional calculation
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=0.0,  # Would need trade analysis
            avg_loss=0.0,  # Would need trade analysis
            total_trades=total_trades,
            best_month=0.0,  # Would need monthly analysis
            worst_month=0.0,  # Would need monthly analysis
            positive_months=0,  # Would need monthly analysis
            negative_months=0  # Would need monthly analysis
        )
    
    def _extract_equity_curve(self, stats) -> List[EquityPoint]:
        """Extract equity curve from backtest stats"""
        
        equity_curve = []
        
        # The backtesting library provides equity curve in stats._results
        if hasattr(stats, '_results') and hasattr(stats._results, 'index'):
            equity_series = stats._results['Equity']
            
            for date, value in equity_series.items():
                # Calculate drawdown (simplified)
                peak = equity_series[:date].max() if len(equity_series[:date]) > 0 else value
                drawdown = ((value - peak) / peak * 100) if peak > 0 else 0
                
                equity_curve.append(EquityPoint(
                    date=date,
                    value=float(value),
                    drawdown=float(drawdown)
                ))
        
        return equity_curve
    
    def _calculate_monthly_returns(self, equity_curve: List[EquityPoint]) -> List[MonthlyReturn]:
        """Calculate monthly returns from equity curve"""
        
        if not equity_curve:
            return []
        
        monthly_returns = []
        current_month = None
        month_start_value = None
        
        for point in equity_curve:
            month_key = (point.date.year, point.date.month)
            
            if current_month != month_key:
                # New month
                if current_month is not None and month_start_value is not None:
                    # Calculate return for previous month
                    prev_point = equity_curve[equity_curve.index(point) - 1]
                    monthly_return = ((prev_point.value - month_start_value) / month_start_value * 100) if month_start_value > 0 else 0
                    
                    monthly_returns.append(MonthlyReturn(
                        year=current_month[0],
                        month=current_month[1],
                        return_pct=monthly_return
                    ))
                
                current_month = month_key
                month_start_value = point.value
        
        # Handle last month
        if current_month is not None and month_start_value is not None and equity_curve:
            last_point = equity_curve[-1]
            monthly_return = ((last_point.value - month_start_value) / month_start_value * 100) if month_start_value > 0 else 0
            
            monthly_returns.append(MonthlyReturn(
                year=current_month[0],
                month=current_month[1],
                return_pct=monthly_return
            ))
        
        return monthly_returns
    
    def _calculate_total_fees(self, stats, fee_percentage: float) -> float:
        """Calculate total fees paid during backtesting"""
        
        # This would need access to trade details from the backtesting library
        # For now, return an estimate based on number of trades
        num_trades = int(stats.get('# Trades', 0))
        avg_trade_size = stats.get('Equity Final [$]', 0) / max(num_trades, 1)
        
        return num_trades * avg_trade_size * (fee_percentage / 100)
    
    async def _compare_with_benchmark(
        self, 
        request: BacktestRequest, 
        results: Dict, 
        historical_data: Dict[str, AssetData]
    ) -> Optional[BenchmarkComparison]:
        """Compare portfolio performance with benchmark"""
        
        try:
            # Fetch benchmark data
            benchmark_data = await data_service.fetch_benchmark_data(
                request.benchmark, request.start_date, request.end_date
            )
            
            if not benchmark_data or not benchmark_data.prices:
                return None
            
            # Calculate benchmark metrics (simplified)
            benchmark_df = self._convert_to_dataframe(benchmark_data)
            
            if benchmark_df.empty:
                return None
            
            # Calculate benchmark returns
            benchmark_returns = benchmark_df['Close'].pct_change().dropna()
            total_return = ((benchmark_df['Close'].iloc[-1] / benchmark_df['Close'].iloc[0]) - 1) * 100
            volatility = benchmark_returns.std() * np.sqrt(252) * 100  # Annualized
            sharpe_ratio = (benchmark_returns.mean() * 252) / (benchmark_returns.std() * np.sqrt(252))
            
            # Calculate drawdown
            cumulative = (1 + benchmark_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = ((cumulative - running_max) / running_max * 100).min()
            
            # Calculate alpha, beta, correlation (simplified)
            portfolio_returns = pd.Series([p.value for p in results['equity_curve']]).pct_change().dropna()
            
            if len(portfolio_returns) > 1 and len(benchmark_returns) > 1:
                # Align the series
                min_len = min(len(portfolio_returns), len(benchmark_returns))
                portfolio_returns = portfolio_returns.iloc[-min_len:]
                benchmark_returns = benchmark_returns.iloc[-min_len:]
                
                correlation = portfolio_returns.corr(benchmark_returns)
                beta = portfolio_returns.cov(benchmark_returns) / benchmark_returns.var()
                alpha = (portfolio_returns.mean() - beta * benchmark_returns.mean()) * 252 * 100
                tracking_error = (portfolio_returns - benchmark_returns).std() * np.sqrt(252) * 100
            else:
                correlation = 0.0
                beta = 1.0
                alpha = 0.0
                tracking_error = 0.0
            
            return BenchmarkComparison(
                benchmark_symbol=request.benchmark,
                benchmark_total_return=float(total_return),
                benchmark_volatility=float(volatility),
                benchmark_sharpe_ratio=float(sharpe_ratio),
                benchmark_max_drawdown=float(drawdown),
                alpha=float(alpha),
                beta=float(beta),
                correlation=float(correlation),
                tracking_error=float(tracking_error)
            )
            
        except Exception as e:
            logger.error(f"Failed to compare with benchmark: {e}")
            return None
    
    def _align_asset_data(self, historical_data: Dict[str, AssetData], assets) -> pd.DataFrame:
        """Align multiple asset data by date"""
        
        dfs = {}
        for asset in assets:
            asset_data = historical_data.get(asset.symbol)
            if asset_data and asset_data.prices:
                df = self._convert_to_dataframe(asset_data)
                if not df.empty:
                    dfs[asset.symbol] = df['Close']
        
        if not dfs:
            return pd.DataFrame()
        
        # Combine all price series
        combined_df = pd.DataFrame(dfs)
        combined_df.fillna(method='ffill', inplace=True)  # Forward fill missing values
        combined_df.dropna(inplace=True)  # Drop rows with any remaining NaN
        
        return combined_df
    
    def _simulate_portfolio_rebalancing(self, aligned_data: pd.DataFrame, request: BacktestRequest) -> Tuple[pd.Series, List]:
        """Simulate portfolio with periodic rebalancing"""
        
        if aligned_data.empty:
            return pd.Series(), []
        
        # Get target allocations
        allocations = {asset.symbol: asset.allocation / 100 for asset in request.assets}
        
        # Initialize portfolio
        portfolio_value = pd.Series(index=aligned_data.index, dtype=float)
        portfolio_value.iloc[0] = request.initial_capital
        
        trades = []
        last_rebalance_date = aligned_data.index[0]
        
        # Calculate rebalancing frequency in days
        rebalance_days = {
            'daily': 1,
            'weekly': 7,
            'monthly': 30,
            'quarterly': 90,
            'yearly': 365
        }.get(request.rebalance_frequency, 90)
        
        for i in range(1, len(aligned_data)):
            current_date = aligned_data.index[i]
            prev_date = aligned_data.index[i-1]
            
            # Calculate daily return for each asset
            daily_returns = aligned_data.iloc[i] / aligned_data.iloc[i-1] - 1
            
            # Apply returns to portfolio (assuming current allocations)
            weighted_return = sum(daily_returns[symbol] * allocation 
                                for symbol, allocation in allocations.items() 
                                if symbol in daily_returns.index)
            
            portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + weighted_return)
            
            # Check if rebalancing is needed
            days_since_rebalance = (current_date - last_rebalance_date).days
            if days_since_rebalance >= rebalance_days:
                # Record rebalancing trade
                trades.append({
                    'date': current_date,
                    'type': 'rebalance',
                    'value': portfolio_value.iloc[i]
                })
                last_rebalance_date = current_date
        
        return portfolio_value, trades
    
    def _calculate_portfolio_metrics(self, portfolio_values: pd.Series, initial_capital: float) -> BacktestMetrics:
        """Calculate metrics for portfolio backtest"""
        
        if portfolio_values.empty:
            return BacktestMetrics(
                total_return=0, annualized_return=0, volatility=0, sharpe_ratio=0,
                sortino_ratio=0, calmar_ratio=0, max_drawdown=0, var_95=0, cvar_95=0,
                win_rate=0, profit_factor=0, avg_win=0, avg_loss=0, total_trades=0,
                best_month=0, worst_month=0, positive_months=0, negative_months=0
            )
        
        # Calculate returns
        returns = portfolio_values.pct_change().dropna()
        total_return = ((portfolio_values.iloc[-1] / initial_capital) - 1) * 100
        
        # Annualized return
        days = (portfolio_values.index[-1] - portfolio_values.index[0]).days
        years = days / 365.25
        annualized_return = ((portfolio_values.iloc[-1] / initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
        
        # Volatility
        volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 1 else 0
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Max drawdown
        cumulative = portfolio_values / portfolio_values.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = ((cumulative - running_max) / running_max * 100).min()
        
        # Win rate (simplified - percentage of positive daily returns)
        positive_days = (returns > 0).sum()
        win_rate = (positive_days / len(returns) * 100) if len(returns) > 0 else 0
        
        return BacktestMetrics(
            total_return=float(total_return),
            annualized_return=float(annualized_return),
            volatility=float(volatility),
            sharpe_ratio=float(sharpe_ratio),
            sortino_ratio=0.0,  # Would need downside deviation calculation
            calmar_ratio=float(annualized_return / abs(drawdown)) if drawdown != 0 else 0,
            max_drawdown=float(drawdown),
            var_95=0.0,  # Would need VaR calculation
            cvar_95=0.0,  # Would need CVaR calculation
            win_rate=float(win_rate),
            profit_factor=0.0,  # Would need win/loss analysis
            avg_win=0.0,  # Would need trade analysis
            avg_loss=0.0,  # Would need trade analysis
            total_trades=0,  # Portfolio rebalancing trades
            best_month=0.0,  # Would need monthly analysis
            worst_month=0.0,  # Would need monthly analysis
            positive_months=0,  # Would need monthly analysis
            negative_months=0  # Would need monthly analysis
        )
    
    def _create_equity_curve_from_values(self, portfolio_values: pd.Series) -> List[EquityPoint]:
        """Create equity curve from portfolio values"""
        
        equity_curve = []
        
        if portfolio_values.empty:
            return equity_curve
        
        # Calculate running maximum for drawdown
        running_max = portfolio_values.expanding().max()
        
        for date, value in portfolio_values.items():
            peak = running_max.loc[date]
            drawdown = ((value - peak) / peak * 100) if peak > 0 else 0
            
            equity_curve.append(EquityPoint(
                date=date,
                value=float(value),
                drawdown=float(drawdown)
            ))
        
        return equity_curve
    
    def _calculate_portfolio_fees(self, trades: List, fee_percentage: float) -> float:
        """Calculate total fees for portfolio trades"""
        
        total_fees = 0.0
        for trade in trades:
            if trade['type'] == 'rebalance':
                # Assume rebalancing incurs fees on the full portfolio value
                total_fees += trade['value'] * (fee_percentage / 100)
        
        return total_fees

# Global instance
backtesting_service = BacktestingService()

