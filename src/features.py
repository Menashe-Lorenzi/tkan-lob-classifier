import pandas as pd
import numpy as np

def calculate_tick_ofi(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates tick-by-tick Order Flow Imbalance (OFI) based on Cont's definition."""
    
    # Calculate price changes
    delta_bid_p = df['bid_price_1'] - df['bid_price_1'].shift(1)
    delta_ask_p = df['ask_price_1'] - df['ask_price_1'].shift(1)
    
    # Calculate bid contribution
    bid_events = np.where(delta_bid_p > 0, df['bid_size_1'],
                 np.where(delta_bid_p < 0, -df['bid_size_1'].shift(1),
                          df['bid_size_1'] - df['bid_size_1'].shift(1)))
                          
    # Calculate ask contribution
    ask_events = np.where(delta_ask_p < 0, df['ask_size_1'],
                 np.where(delta_ask_p > 0, -df['ask_size_1'].shift(1),
                          df['ask_size_1'] - df['ask_size_1'].shift(1)))
                          
    # Combine both sides to create the raw imbalance event
    df['e_n'] = bid_events - ask_events
    
    return df

def generate_features(df: pd.DataFrame, resample_freq: str = '1s') -> pd.DataFrame:
    """Processes LOB data, calculates OFI, resamples, and adds microstructural features."""
    
    # Ensure the index is a Datetime object
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)

    # Calculate raw tick events
    df = calculate_tick_ofi(df)

    # Define aggregation rules for resampling
    aggregation_rules = {
        'e_n': 'sum',               
        'ask_price_1': 'last',      
        'bid_price_1': 'last',
        'ask_price_2': 'last',
        'bid_price_2': 'last',
        'ask_size_1': 'mean',       
        'bid_size_1': 'mean',
        'ask_size_2': 'mean',
        'bid_size_2': 'mean'
    }

    # Apply time aggregation
    df_resampled = df.resample(resample_freq).agg(aggregation_rules)
    df_resampled.dropna(inplace=True)
    df_resampled.rename(columns={'e_n': 'ofi'}, inplace=True)

    # Engineer derived features for the model
    # Mid-Price: average between top ask and top bid
    df_resampled['mid_price'] = (df_resampled['ask_price_1'] + df_resampled['bid_price_1']) / 2.0

    # Spread: difference between top ask and top bid
    df_resampled['spread'] = df_resampled['ask_price_1'] - df_resampled['bid_price_1']

    return df_resampled


def engineer_lob_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes an aggregated LOB DataFrame and generates advanced stationary features.
    Removes raw absolute prices to prevent non-stationarity in models.
    """
    df = df.copy()
    
    # Calculate relative price distances to maintain stationarity
    df['ask_dist_1'] = df['ask_price_1'] - df['mid_price']
    df['bid_dist_1'] = df['mid_price'] - df['bid_price_1']
    df['ask_dist_2'] = df['ask_price_2'] - df['mid_price']
    df['bid_dist_2'] = df['mid_price'] - df['bid_price_2']
    
    # Calculate volume imbalance with a small constant to prevent division by zero
    df['vol_imb_1'] = (df['bid_size_1'] - df['ask_size_1']) / (df['bid_size_1'] + df['ask_size_1'] + 1e-8)
    df['vol_imb_2'] = (df['bid_size_2'] - df['ask_size_2']) / (df['bid_size_2'] + df['ask_size_2'] + 1e-8)
    
    # Calculate micro-price and price pressure
    micro_price = (df['bid_price_1'] * df['ask_size_1'] + df['ask_price_1'] * df['bid_size_1']) / (df['bid_size_1'] + df['ask_size_1'] + 1e-8)
    df['price_pressure'] = micro_price - df['mid_price']
    
    # Calculate macro context features using a rolling window
    df['ofi_momentum'] = df['ofi'].rolling(window=60).mean()
    
    # Calculate volatility based on mid-price returns
    returns = df['mid_price'].pct_change()
    df['volatility'] = returns.rolling(window=60).std()
    
    # Drop absolute raw prices since the model should rely on stationary features
    cols_to_drop = [
        'ask_price_1', 'bid_price_1', 'ask_price_2', 'bid_price_2', 'mid_price'
    ]
    df.drop(columns=cols_to_drop, inplace=True)
    
    # Drop NaN rows created by the rolling windows
    df.dropna(inplace=True)
    
    return df