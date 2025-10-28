import pandas as pd
import numpy as np
import logging
from .data_preprocessor import create_basic_features

def create_advanced_features(df):
    """Create features using our basic feature engineering"""
    return create_basic_features(df)

def prepare_data(df, lookback_days=[3, 5, 10, 20]):
    """Prepare data for modeling - FIXED to preserve original columns"""
    logger = logging.getLogger('TradingBot')
    
    # Remove duplicate columns first
    df = df.loc[:, ~df.columns.duplicated()].copy()
    
    # Use basic features
    df = create_basic_features(df)
    
    # Remove duplicates again after feature creation
    df = df.loc[:, ~df.columns.duplicated()].copy()
    
    # Create lookback features
    for days in lookback_days:
        df[f'Returns_{days}d'] = df['Close'].pct_change(days)
        df[f'Volatility_{days}d'] = df['Returns'].rolling(days).std()
        df[f'MA_{days}d'] = df['Close'].rolling(days).mean()
    
    # Target variable (predict next day return direction)
    df['Target'] = (df['Returns'].shift(-1) > 0).astype(int)
    
    # Define only features that definitely exist
    base_features = [f'Returns_{d}d' for d in lookback_days] + \
                   [f'Volatility_{d}d' for d in lookback_days] + \
                   [f'MA_{d}d' for d in lookback_days]
    
    # Only use features we know exist
    available_features = []
    all_possible_features = base_features + [
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
        'SMA_20', 'EMA_20', 'SMA_50', 'EMA_50',
        'Volume_Ratio', 'Volume_Spike',
        'RSI_30', 'RSI_70'
    ]
    
    for feature in all_possible_features:
        if feature in df.columns:
            available_features.append(feature)
    
    missing_features = set(all_possible_features) - set(available_features)
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")
    
    # Handle missing values - BUT preserve original price columns for trading!
    # We need to keep Close, Open, High, Low, Volume for the trading simulation
    essential_columns = ['Close', 'Open', 'High', 'Low', 'Volume', 'Target'] + available_features
    df_clean = df[essential_columns].copy()
    df_clean = df_clean.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    if len(df_clean) < 50:
        raise ValueError(f"Insufficient data after cleaning: {len(df_clean)} samples")
    
    # For model training, we only use the feature columns
    X = df_clean[available_features].values
    y = df_clean['Target'].values
    
    logger.info(f"Prepared data: {len(X)} samples, {len(available_features)} features")
    logger.info(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")
    
    return X, y, df_clean, available_features