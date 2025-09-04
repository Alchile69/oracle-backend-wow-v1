"""
FastAPI router for backtesting endpoints
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
import asyncio
import logging
from datetime import datetime, timedelta
import uuid
import math
import json

# Firebase imports for persistence
from firebase_config import db

# Import models and services
from models.backtest import (
    BacktestRequest, BacktestResults, BacktestError, BacktestStatus,
    StrategyType, RebalanceFrequency, AssetData
)
from services.backtesting_service import backtesting_service

logger = logging.getLogger(__name__)

# Check if Firestore is available
FIRESTORE_AVAILABLE = db is not None
if FIRESTORE_AVAILABLE:
    logger.info("✅ Firestore client available from Railway config")
else:
    logger.warning("⚠️ Firestore not available, using in-memory storage")

router = APIRouter(prefix="/api/backtest", tags=["backtesting"])

# Fallback in-memory storage if Firestore is not available
backtest_status_store: Dict[str, BacktestStatus] = {}
backtest_results_store: Dict[str, BacktestResults] = {}

def _clean_float_values(obj):
    """Clean float values to ensure JSON compliance"""
    if isinstance(obj, dict):
        return {k: _clean_float_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_float_values(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj):
            return 0.0
        elif math.isinf(obj):
            return 999999.0 if obj > 0 else -999999.0
        return obj
    return obj

# Firestore persistence functions
async def _save_backtest_status(request_id: str, status: BacktestStatus):
    """Save backtest status to Firestore or fallback to memory"""
    if FIRESTORE_AVAILABLE and db:
        try:
            doc_ref = db.collection('backtests').document(request_id)
            status_dict = status.dict()
            status_dict['updated_at'] = firestore.SERVER_TIMESTAMP
            doc_ref.set(status_dict, merge=True)
            logger.info(f"Saved status for {request_id} to Firestore")
        except Exception as e:
            logger.error(f"Failed to save status to Firestore: {e}")
            backtest_status_store[request_id] = status
    else:
        backtest_status_store[request_id] = status

async def _get_backtest_status(request_id: str) -> Optional[BacktestStatus]:
    """Get backtest status from Firestore or fallback to memory"""
    if FIRESTORE_AVAILABLE and db:
        try:
            doc_ref = db.collection('backtests').document(request_id)
            doc = doc_ref.get()
            if doc.exists:
                data = doc.to_dict()
                return BacktestStatus(**data)
        except Exception as e:
            logger.error(f"Failed to get status from Firestore: {e}")
    
    return backtest_status_store.get(request_id)

async def _save_backtest_results(request_id: str, results: BacktestResults):
    """Save backtest results to Firestore or fallback to memory"""
    if FIRESTORE_AVAILABLE and db:
        try:
            doc_ref = db.collection('backtests').document(request_id)
            results_dict = _clean_float_values(results.dict())
            results_dict['completed_at'] = firestore.SERVER_TIMESTAMP
            doc_ref.set({'results': results_dict}, merge=True)
            logger.info(f"Saved results for {request_id} to Firestore")
        except Exception as e:
            logger.error(f"Failed to save results to Firestore: {e}")
            backtest_results_store[request_id] = results
    else:
        backtest_results_store[request_id] = results

async def _get_backtest_results(request_id: str) -> Optional[BacktestResults]:
    """Get backtest results from Firestore or fallback to memory"""
    if FIRESTORE_AVAILABLE and db:
        try:
            doc_ref = db.collection('backtests').document(request_id)
            doc = doc_ref.get()
            if doc.exists:
                data = doc.to_dict()
                if 'results' in data:
                    return BacktestResults(**data['results'])
        except Exception as e:
            logger.error(f"Failed to get results from Firestore: {e}")
    
    return backtest_results_store.get(request_id)

async def _get_backtest_history() -> List[Dict]:
    """Get backtest history from Firestore or fallback to memory"""
    history = []
    
    if FIRESTORE_AVAILABLE and db:
        try:
            docs = db.collection('backtests').order_by('created_at', direction=firestore.Query.DESCENDING).limit(50).stream()
            for doc in docs:
                data = doc.to_dict()
                history.append({
                    'request_id': doc.id,
                    'status': data.get('status', 'unknown'),
                    'created_at': data.get('created_at'),
                    'progress': data.get('progress', 0)
                })
        except Exception as e:
            logger.error(f"Failed to get history from Firestore: {e}")
    
    # Fallback to memory
    if not history:
        for request_id, status in backtest_status_store.items():
            history.append({
                'request_id': request_id,
                'status': status.status,
                'created_at': status.created_at,
                'progress': status.progress
            })
    
    return history

@router.post("/run", response_model=Dict[str, str])
async def run_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks
):
    """
    Start a new backtest run
    
    Returns immediately with a request_id for status tracking
    """
    try:
        # Generate unique request ID
        request_id = f"bt_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}"
        
        # Validate request
        validation_errors = _validate_backtest_request(request)
        if validation_errors:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid request: {', '.join(validation_errors)}"
            )
        
        # Create initial status
        status = BacktestStatus(
            request_id=request_id,
            status="pending",
            progress=0,
            message="Backtest queued for execution",
            created_at=datetime.now()
        )
        
        # Save initial status to Firestore
        await _save_backtest_status(request_id, status)
        
        # Start backtest in background
        background_tasks.add_task(_run_backtest_background, request_id, request)
        
        logger.info(f"Started backtest {request_id} for {len(request.assets)} assets")
        
        return {
            "request_id": request_id,
            "status": "pending",
            "message": "Backtest started successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start backtest: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start backtest: {str(e)}")

@router.get("/status/{request_id}", response_model=BacktestStatus)
async def get_backtest_status(request_id: str):
    """Get the status of a running backtest"""
    
    status = await _get_backtest_status(request_id)
    if not status:
        raise HTTPException(status_code=404, detail="Backtest not found")
    
    return status

@router.get("/results/{request_id}", response_model=BacktestResults)
async def get_backtest_results(request_id: str):
    """Get the results of a completed backtest"""
    
    status = await _get_backtest_status(request_id)
    if not status:
        raise HTTPException(status_code=404, detail="Backtest not found")
    
    if status.status != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Backtest not completed yet. Current status: {status.status}"
        )
    
    results = await _get_backtest_results(request_id)
    if not results:
        raise HTTPException(status_code=404, detail="Backtest results not found")
    
    return results

@router.delete("/cancel/{request_id}")
async def cancel_backtest(request_id: str):
    """Cancel a running backtest"""
    
    if request_id not in backtest_status_store:
        raise HTTPException(status_code=404, detail="Backtest not found")
    
    status = backtest_status_store[request_id]
    
    if status.status in ["completed", "failed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel backtest with status: {status.status}"
        )
    
    # Update status to cancelled
    status.status = "cancelled"
    status.message = "Backtest cancelled by user"
    status.updated_at = datetime.now()
    
    return {"message": "Backtest cancelled successfully"}

@router.get("/history", response_model=List[BacktestStatus])
async def get_backtest_history(limit: int = 10):
    """Get history of recent backtests"""
    
    # Sort by creation time, most recent first
    sorted_statuses = sorted(
        backtest_status_store.values(),
        key=lambda x: x.created_at,
        reverse=True
    )
    
    return sorted_statuses[:limit]

@router.get("/strategies", response_model=List[Dict[str, str]])
async def get_available_strategies():
    """Get list of available trading strategies"""
    
    strategies = [
        {
            "id": StrategyType.BUY_AND_HOLD,
            "name": "Buy and Hold",
            "description": "Simple buy and hold strategy - buy at start and hold until end"
        },
        {
            "id": StrategyType.SMA_CROSSOVER,
            "name": "SMA Crossover",
            "description": "Simple Moving Average crossover strategy - buy when short SMA crosses above long SMA"
        },
        {
            "id": StrategyType.RSI_OVERSOLD,
            "name": "RSI Oversold/Overbought",
            "description": "RSI-based strategy - buy when oversold, sell when overbought"
        },
        {
            "id": StrategyType.BOLLINGER_BANDS,
            "name": "Bollinger Bands",
            "description": "Mean reversion strategy using Bollinger Bands"
        }
    ]
    
    return strategies

@router.post("/quick-run", response_model=BacktestResults)
async def run_quick_backtest(request: BacktestRequest):
    """
    Run a quick backtest synchronously (for simple/fast backtests)
    
    Use this for single assets or short time periods
    """
    try:
        # Validate request
        validation_errors = _validate_backtest_request(request)
        if validation_errors:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid request: {', '.join(validation_errors)}"
            )
        
        # Check if this is suitable for quick run
        days = (request.end_date - request.start_date).days
        if days > 365 or len(request.assets) > 3:
            raise HTTPException(
                status_code=400,
                detail="Quick backtest is limited to 1 year and 3 assets maximum. Use /run for larger backtests."
            )
        
        # Run backtest
        logger.info(f"Running quick backtest for {len(request.assets)} assets over {days} days")
        results = await backtesting_service.run_backtest(request)
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quick backtest failed: {e}")
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")

@router.get("/validate", response_model=Dict[str, Any])
async def validate_backtest_config(
    symbols: str,  # Comma-separated symbols
    start_date: str,  # YYYY-MM-DD
    end_date: str,  # YYYY-MM-DD
):
    """
    Validate a backtest configuration before running
    
    Checks data availability, date ranges, etc.
    """
    try:
        from datetime import datetime
        from ..services.data_service import data_service
        
        # Parse parameters
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Validate date range
        if start_dt >= end_dt:
            return {
                "valid": False,
                "errors": ["Start date must be before end date"]
            }
        
        if end_dt > datetime.now():
            return {
                "valid": False,
                "errors": ["End date cannot be in the future"]
            }
        
        # Check data availability
        historical_data = await data_service.fetch_historical_data(
            symbol_list, start_dt, end_dt
        )
        
        errors = []
        warnings = []
        
        for symbol in symbol_list:
            asset_data = historical_data.get(symbol)
            if not asset_data or not asset_data.prices:
                errors.append(f"No data available for {symbol}")
            else:
                is_valid, data_warnings = data_service.validate_data_quality(asset_data)
                if not is_valid:
                    errors.append(f"Insufficient data quality for {symbol}")
                warnings.extend(data_warnings)
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "data_points": {
                symbol: len(historical_data.get(symbol, AssetData(symbol=symbol, prices=[])).prices)
                for symbol in symbol_list
            }
        }
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return {
            "valid": False,
            "errors": [f"Validation error: {str(e)}"]
        }

async def _run_backtest_background(request_id: str, request: BacktestRequest):
    """Run backtest in background task with Firestore persistence"""
    
    try:
        # Get current status
        status = await _get_backtest_status(request_id)
        if not status:
            logger.error(f"Status not found for {request_id}")
            return
        
        # Update status to running
        status.status = "running"
        status.progress = 10
        status.message = "Fetching historical data..."
        status.updated_at = datetime.now()
        await _save_backtest_status(request_id, status)
        
        logger.info(f"Starting backtest {request_id} with {len(request.assets)} assets")
        
        # Update progress
        status.progress = 30
        status.message = "Running backtest algorithms..."
        await _save_backtest_status(request_id, status)
        
        # Run the actual backtest
        results = await backtesting_service.run_backtest(request)
        
        # Update progress
        status.progress = 80
        status.message = "Processing results..."
        await _save_backtest_status(request_id, status)
        
        # Clean float values to ensure JSON compliance
        cleaned_results = _clean_float_values(results.dict())
        
        # Convert back to BacktestResults object
        cleaned_results_obj = BacktestResults(**cleaned_results)
        
        # Save results to Firestore
        await _save_backtest_results(request_id, cleaned_results_obj)
        
        # Update status to completed
        status.status = "completed"
        status.progress = 100
        status.message = "Backtest completed successfully"
        status.updated_at = datetime.now()
        await _save_backtest_status(request_id, status)
        
        logger.info(f"Backtest {request_id} completed successfully - Results saved to Firestore")
        
    except Exception as e:
        # Update status to failed
        status = await _get_backtest_status(request_id)
        if status:
            status.status = "failed"
            status.progress = 0
            status.message = f"Backtest failed: {str(e)}"
            status.updated_at = datetime.now()
            await _save_backtest_status(request_id, status)
        
        logger.error(f"Backtest {request_id} failed: {e}", exc_info=True)

def _validate_backtest_request(request: BacktestRequest) -> List[str]:
    """Validate backtest request parameters"""
    
    errors = []
    
    # Validate assets
    if not request.assets:
        errors.append("At least one asset is required")
    
    total_allocation = sum(asset.allocation for asset in request.assets)
    if abs(total_allocation - 100.0) > 0.01:  # Allow small floating point errors
        errors.append(f"Asset allocations must sum to 100%, got {total_allocation}%")
    
    for asset in request.assets:
        if not asset.symbol or not asset.symbol.strip():
            errors.append("Asset symbol cannot be empty")
        if asset.allocation <= 0:
            errors.append(f"Asset allocation must be positive, got {asset.allocation}% for {asset.symbol}")
    
    # Validate dates
    if request.start_date >= request.end_date:
        errors.append("Start date must be before end date")
    
    if request.end_date > datetime.now().date():
        errors.append("End date cannot be in the future")
    
    # Validate date range (not too long for performance)
    days = (request.end_date - request.start_date).days
    if days > 3650:  # 10 years
        errors.append("Backtest period cannot exceed 10 years")
    
    if days < 30:  # Minimum 30 days
        errors.append("Backtest period must be at least 30 days")
    
    # Validate capital
    if request.initial_capital < 1000:
        errors.append("Initial capital must be at least $1,000")
    
    # Validate fees
    if request.transaction_fees < 0 or request.transaction_fees > 5:
        errors.append("Transaction fees must be between 0% and 5%")
    
    if request.slippage < 0 or request.slippage > 2:
        errors.append("Slippage must be between 0% and 2%")
    
    # Validate strategy parameters
    if request.strategy_type == StrategyType.SMA_CROSSOVER:
        if request.sma_short >= request.sma_long:
            errors.append("Short SMA period must be less than long SMA period")
    
    return errors

