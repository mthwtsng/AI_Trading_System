import pandas as pd
import numpy as np
import logging
from sklearn.feature_selection import SelectKBest, f_classif

def _ensure_series(df, col, logger):
    """Return a single Series for a column, even if duplicate-named columns exist."""
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        logger.warning(f"Column '{col}' has duplicates; using the first occurrence")
        obj = obj.iloc[:, 0]
    return obj.astype(float)

def create_basic_features(df):
    """Create basic features that don't rely on external libraries"""
    logger = logging.getLogger('TradingBot')
    logger.info("Creating basic features...")
    
    # Work on a copy
    df = df.copy()

    # Drop duplicate-named columns to avoid Series->DataFrame surprises
    if df.columns.has_duplicates:
        logger.warning(f"Found duplicate columns, deduplicating: {df.columns[df.columns.duplicated()].tolist()}")
        df = df.loc[:, ~df.columns.duplicated()].copy()
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Coerce base inputs to Series
    close = _ensure_series(df, 'Close', logger)
    volume = _ensure_series(df, 'Volume', logger)
    
    # Basic price features
    df['Returns'] = close.pct_change()
    df['Log_Returns'] = np.log(close / close.shift(1))
    
    # Simple moving averages
    for window in [5, 10, 20, 50]:
        df[f'SMA_{window}'] = close.rolling(window).mean()
        df[f'EMA_{window}'] = close.ewm(span=window).mean()
    
    # Close rank (use Close-only rolling min/max to avoid helper column collisions)
    for window in [5, 10, 20]:
        roll_max = close.rolling(window).max()
        roll_min = close.rolling(window).min()
        denom = (roll_max - roll_min).replace(0, np.nan)
        close_rank = (close - roll_min) / denom
        close_rank = close_rank.replace([np.inf, -np.inf], np.nan).fillna(0.5)
        df[f'Close_Rank_{window}'] = close_rank.astype(float)
    
    # Volume features (ensure Series, avoid DataFrame in fillna)
    vol_sma_20 = volume.rolling(20).mean()
    vol_sma_20 = vol_sma_20.where(vol_sma_20.notna(), volume)  # fillna with Series safely
    df['Volume_SMA_20'] = vol_sma_20
    df['Volume_Ratio'] = (volume / vol_sma_20).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    df['Volume_Spike'] = (df['Volume_Ratio'] > 2).astype(int)
    
    # Volatility
    for window in [5, 10, 20]:
        df[f'Volatility_{window}'] = df['Returns'].rolling(window).std()
    
    # Price momentum
    for period in [1, 3, 5, 10]:
        df[f'Momentum_{period}'] = close.pct_change(period)
    
    # RSI calculation (manual)
    df['RSI'] = calculate_rsi_manual(close)
    df['RSI_30'] = (df['RSI'] < 30).astype(int)
    df['RSI_70'] = (df['RSI'] > 70).astype(int)
    
    # MACD calculation (manual)
    macd, macd_sig = calculate_macd_manual(close)
    df['MACD'], df['MACD_Signal'] = macd, macd_sig
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Fill NaN values
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    logger.info(f"Created {len([c for c in df.columns if c not in ['Open','High','Low','Close','Volume']])} basic features")
    return df

def calculate_rsi_manual(close, window=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    loss = loss.replace(0, 0.0001)
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_macd_manual(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast).mean()
    ema_slow = close.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    return macd.fillna(0), macd_signal.fillna(0)

def select_best_features(df, target_col='Target', k=20):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != target_col]
    if len(numeric_cols) == 0:
        return []
    X = df[numeric_cols].fillna(0)
    y = df[target_col]
    selector = SelectKBest(score_func=f_classif, k=min(k, len(numeric_cols)))
    selector.fit(X, y)
    return X.columns[selector.get_support()].tolist()