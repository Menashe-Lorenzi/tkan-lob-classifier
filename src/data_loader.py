import pandas as pd
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Assigns correct column names for Limit Order Book data."""
    n = df.shape[1]
    
    # Properly check if the number of columns is a multiple of 4
    if n % 4 != 0:
        raise ValueError(f"Invalid number of columns: {n}. Must be a multiple of 4.")

    names = []
    # LOBSTER data structure: ask price, ask size, bid price, bid size
    for i in range(1, (n // 4) + 1):
        names.extend([f"ask_price_{i}", f"ask_size_{i}", f"bid_price_{i}", f"bid_size_{i}"])

    df.columns = names
    return df

def load_lob_data(stock_ticker: str = "AAPL", date: str = '2025-09-22') -> pd.DataFrame:
    """Loads message and orderbook data for a single day and returns the timestamped orderbook."""
    base_path = PROJECT_ROOT / 'data' / 'M7_LOB' / stock_ticker
    
    # Ensure date is properly formatted as a string for the file path
    date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
    
    msg_file = base_path / f'{stock_ticker}_{date_str}_34200000_57600000_message_2.csv'
    ord_book_file = base_path / f'{stock_ticker}_{date_str}_34200000_57600000_orderbook_2.csv'
    
    # Check if files exist to avoid crashing on weekends/holidays
    if not msg_file.exists() or not ord_book_file.exists():
        print(f"Data files missing for {stock_ticker} on {date_str}. Skipping.")
        return pd.DataFrame() 
        
    # Read CSVs. Assuming LOBSTER format, there are usually no headers in the raw files
    msg_df = pd.read_csv(msg_file, header=None) 
    ord_book_df = pd.read_csv(ord_book_file, header=None)
    
    # Map columns, handling potential trailing null columns common in LOBSTER
    msg_cols = ["time", "type", "id", "size", "price", "direction", "null"]
    msg_df.columns = msg_cols[:msg_df.shape[1]] 
    
    ord_book_df = order_columns(ord_book_df)

    # Create proper datetime index
    base_date = pd.Timestamp(date_str)
    time_delta = pd.to_timedelta(msg_df['time'], unit='s')
    ord_book_df["Time"] = base_date + time_delta
    ord_book_df.set_index("Time", inplace=True)

    return ord_book_df

def load_lob_data_range(stock_ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Loads and concatenates LOB data for a specific stock across a date range."""
    # Generate a list of business days between the start and end dates
    dates = pd.date_range(start=start_date, end=end_date, freq='B') 
    
    daily_dfs = []
    
    for date in dates:
        # Load data for each day
        daily_df = load_lob_data(stock_ticker, date=date)
        
        # If the dataframe is not empty (meaning the file existed), add it to our list
        if not daily_df.empty:
            daily_dfs.append(daily_df)
            
    if not daily_dfs:
        print(f"No data found for {stock_ticker} between {start_date} and {end_date}.")
        return pd.DataFrame()
        
    # Concatenate all the daily dataframes into one large dataframe
    combined_df = pd.concat(daily_dfs)
    return combined_df