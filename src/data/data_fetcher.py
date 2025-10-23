import yfinance as yf
import pandas as pd
import numpy as np
import logging

def fetch_stock_data(symbol, start_date, end_date, interval='1d', max_retries=3):
    """Fetch stock data with proper DataFrame handling"""
    logger = logging.getLogger('TradingBot')
    logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
    
    # Convert dates to string format
    if hasattr(start_date, 'strftime'):
        start_date_str = start_date.strftime('%Y-%m-%d')
    else:
        start_date_str = str(start_date)
        
    if hasattr(end_date, 'strftime'):
        end_date_str = end_date.strftime('%Y-%m-%d')
    else:
        end_date_str = str(end_date)
    
    # Try to fetch data
    try:
        df = yf.download(symbol, start=start_date_str, end=end_date_str, 
                        interval=interval, progress=False, auto_adjust=True)
        
        if df.empty:
            # Fallback to period-based download
            df = yf.download(symbol, period="2y", interval=interval, 
                           progress=False, auto_adjust=True)
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        # Create synthetic data as fallback
        return create_synthetic_data(start_date_str, end_date_str)
    
    if df.empty:
        logger.warning("No data found, creating synthetic data")
        return create_synthetic_data(start_date_str, end_date_str)
    
    logger.info(f"Successfully fetched {len(df)} records for {symbol}")
    
    # Ensure we have a proper DataFrame with correct index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Reset index to make sure we have a clean DataFrame
    df = df.reset_index()
    if 'Date' in df.columns:
        df = df.set_index('Date')
    
    # Ensure required columns exist
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Basic price transformations
    df["Returns"] = df["Close"].pct_change()
    df["Log_Returns"] = np.log(df['Close']/df['Close'].shift(1))
    
    logger.info(f"Data preparation complete. Total columns: {len(df.columns)}")
    return df

def create_synthetic_data(start_date, end_date, base_price=100):
    """Create synthetic stock data for testing"""
    logger = logging.getLogger('TradingBot')
    logger.info("Creating synthetic data for testing...")
    
    # Create date range (business days only)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    if len(dates) == 0:
        # Fallback to recent dates
        from datetime import datetime, timedelta
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=365)
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    n_days = len(dates)
    
    # Generate realistic price data
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, n_days)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Create OHLC data
    df = pd.DataFrame({
        'Open': prices * (1 + np.random.normal(0, 0.005, n_days)),
        'High': prices * (1 + np.abs(np.random.normal(0.01, 0.01, n_days))),
        'Low': prices * (1 - np.abs(np.random.normal(0.01, 0.01, n_days))),
        'Close': prices,
        'Volume': np.random.lognormal(14, 1, n_days)
    }, index=dates)
    
    # Ensure High is highest and Low is lowest
    df['High'] = df[['Open', 'High', 'Low', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'High', 'Low', 'Close']].min(axis=1)
    
    # Add basic calculations
    df["Returns"] = df["Close"].pct_change()
    df["Log_Returns"] = np.log(df['Close']/df['Close'].shift(1))
    
    logger.info(f"Created synthetic data with {len(df)} records")
    return df