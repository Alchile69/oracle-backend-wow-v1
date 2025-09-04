"""
Models for backtesting requests and responses
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, date
from enum import Enum

class RebalanceFrequency(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"

class StrategyType(str, Enum):
    SMA_CROSSOVER = "sma_crossover"
    EMA_CROSSOVER = "ema_crossover"
    RSI_OVERSOLD = "rsi_oversold"
    BOLLINGER_BANDS = "bollinger_bands"
    BUY_AND_HOLD = "buy_and_hold"

class AssetAllocation(BaseModel):
    symbol: str = Field(..., description="Asset symbol (e.g., AAPL)")
    allocation: float = Field(..., ge=0, le=100, description="Allocation percentage (0-100)")

class BacktestRequest(BaseModel):
    # Portfolio configuration
    assets: List[AssetAllocation] = Field(..., description="List of assets and their allocations")
    initial_capital: float = Field(100000, ge=1000, description="Initial capital in USD")
    
    # Date range
    start_date: date = Field(..., description="Backtest start date")
    end_date: date = Field(..., description="Backtest end date")
    
    # Strategy parameters
    strategy_type: StrategyType = Field(StrategyType.BUY_AND_HOLD, description="Trading strategy to use")
    rebalance_frequency: RebalanceFrequency = Field(RebalanceFrequency.QUARTERLY, description="Portfolio rebalancing frequency")
    
    # Trading costs
    transaction_fees: float = Field(0.1, ge=0, le=5, description="Transaction fees percentage")
    slippage: float = Field(0.05, ge=0, le=2, description="Slippage percentage")
    
    # Options
    dividend_reinvestment: bool = Field(True, description="Reinvest dividends")
    benchmark: str = Field("SPY", description="Benchmark symbol for comparison")
    
    # Strategy-specific parameters
    sma_short: int = Field(10, ge=1, le=200, description="Short SMA period")
    sma_long: int = Field(20, ge=1, le=200, description="Long SMA period")
    rsi_period: int = Field(14, ge=1, le=100, description="RSI period")
    rsi_oversold: float = Field(30, ge=0, le=100, description="RSI oversold threshold")
    rsi_overbought: float = Field(70, ge=0, le=100, description="RSI overbought threshold")

class BacktestMetrics(BaseModel):
    # Performance metrics
    total_return: float = Field(..., description="Total return percentage")
    annualized_return: float = Field(..., description="Annualized return percentage")
    volatility: float = Field(..., description="Annualized volatility percentage")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    calmar_ratio: float = Field(..., description="Calmar ratio")
    
    # Risk metrics
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")
    var_95: float = Field(..., description="Value at Risk (95%)")
    cvar_95: float = Field(..., description="Conditional Value at Risk (95%)")
    
    # Trading metrics
    win_rate: float = Field(..., description="Win rate percentage")
    profit_factor: float = Field(..., description="Profit factor")
    avg_win: float = Field(..., description="Average winning trade")
    avg_loss: float = Field(..., description="Average losing trade")
    total_trades: int = Field(..., description="Total number of trades")
    
    # Time-based metrics
    best_month: float = Field(..., description="Best monthly return")
    worst_month: float = Field(..., description="Worst monthly return")
    positive_months: int = Field(..., description="Number of positive months")
    negative_months: int = Field(..., description="Number of negative months")

class BenchmarkComparison(BaseModel):
    benchmark_symbol: str
    benchmark_total_return: float
    benchmark_volatility: float
    benchmark_sharpe_ratio: float
    benchmark_max_drawdown: float
    alpha: float = Field(..., description="Portfolio alpha vs benchmark")
    beta: float = Field(..., description="Portfolio beta vs benchmark")
    correlation: float = Field(..., description="Correlation with benchmark")
    tracking_error: float = Field(..., description="Tracking error vs benchmark")

class EquityPoint(BaseModel):
    date: datetime
    value: float
    drawdown: float = Field(..., description="Drawdown percentage at this point")

class MonthlyReturn(BaseModel):
    year: int
    month: int
    return_pct: float

class BacktestResults(BaseModel):
    # Request info
    request_id: str = Field(..., description="Unique request identifier")
    created_at: datetime = Field(default_factory=datetime.now)
    config: BacktestRequest
    
    # Results
    metrics: BacktestMetrics
    benchmark_comparison: Optional[BenchmarkComparison] = None
    
    # Time series data
    equity_curve: List[EquityPoint] = Field(..., description="Portfolio value over time")
    monthly_returns: List[MonthlyReturn] = Field(..., description="Monthly returns breakdown")
    
    # Additional data
    final_portfolio_value: float
    total_fees_paid: float
    execution_time_seconds: float
    
    # Warnings and notes
    warnings: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)

class BacktestError(BaseModel):
    error_type: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class BacktestStatus(BaseModel):
    request_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: float = Field(0, ge=0, le=100)  # Progress percentage
    message: str = ""
    created_at: datetime
    updated_at: datetime = Field(default_factory=datetime.now)

# Historical data models
class PriceData(BaseModel):
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float] = None

class AssetData(BaseModel):
    symbol: str
    prices: List[PriceData]
    dividends: Optional[List[Dict[str, Any]]] = None
    splits: Optional[List[Dict[str, Any]]] = None

