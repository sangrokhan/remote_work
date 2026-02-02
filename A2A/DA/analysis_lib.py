
import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from typing import List, Callable, Union, Dict

def load_data(filepath: str, file_type: str = 'parquet') -> pd.DataFrame:
    """
    Loads data from a file.
    """
    try:
        if file_type == 'parquet':
            return pd.read_parquet(filepath)
        elif file_type == 'csv':
            return pd.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    except FileNotFoundError:
        print(f"Warning: File {filepath} not found.")
        return pd.DataFrame()

def filter_data_by_cell(df: pd.DataFrame, cell_col: str, target_cell_id: Union[str, int]) -> pd.DataFrame:
    """
    Filters the DataFrame for a specific Cell ID.
    """
    if df.empty:
        return df
    return df[df[cell_col] == target_cell_id].copy()

def aggregate_time_rows(df: pd.DataFrame, time_col: str, agg_func: str = 'mean') -> pd.DataFrame:
    """
    Aggregates multiple rows with the same timestamp into a single row.
    Useful if there are sub-readings for the same time.
    """
    if df.empty:
        return df
    
    # Select only numeric columns for aggregation to avoid errors
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Ensure time_col is preserved if it's numeric, or used as grouper
    if time_col not in numeric_cols and time_col in df.columns:
        # If time_col is not numeric (e.g. datetime), it will be handled by groupby
        pass
        
    grouped = df.groupby(time_col)[numeric_cols]
    
    if agg_func == 'mean':
        return grouped.mean().reset_index()
    elif agg_func == 'sum':
        return grouped.sum().reset_index()
    elif agg_func == 'max':
        return grouped.max().reset_index()
    elif agg_func == 'min':
        return grouped.min().reset_index()
    else:
        raise ValueError(f"Unsupported aggregation function: {agg_func}")

def get_row_distribution(row: pd.Series, target_cols: List[str]) -> np.ndarray:
    """
    Extracts values from specific columns to form a distribution array.
    Normalizes the distribution to sum to 1 for probability density metrics like JS Div.
    """
    data = row[target_cols].values.astype(float)
    # Handle NaNs
    data = np.nan_to_num(data)
    
    # Normalize for probability distribution (required for JS divergence)
    if np.sum(data) > 0:
        return data / np.sum(data)
    return data

def calculate_distribution_change(
    current_dist: np.ndarray, 
    baseline_dist: np.ndarray, 
    method: str = 'jensenshannon'
) -> float:
    """
    Calculates the difference between two distributions.
    """
    if np.sum(current_dist) == 0 or np.sum(baseline_dist) == 0:
        return 0.0

    if method == 'jensenshannon':
        return jensenshannon(current_dist, baseline_dist)
    elif method == 'wasserstein':
        return wasserstein_distance(current_dist, baseline_dist)
    elif method == 'euclidean':
        return np.linalg.norm(current_dist - baseline_dist)
    else:
        raise ValueError(f"Unknown analysis method: {method}")

class AnalysisExecutor:
    """
    Helper class to manage stateful analysis if needed (e.g., keeping a running baseline).
    """
    def __init__(self, method: str = 'jensenshannon'):
        self.method = method
        self.baseline = None

    def analyze(self, current_dist: np.ndarray) -> float:
        if self.baseline is None:
            self.baseline = current_dist
            return 0.0
        
        score = calculate_distribution_change(current_dist, self.baseline, self.method)
        # Optional: Update baseline? For now, we compare to fixed previous or running.
        # Let's assume sequential comparison (t vs t-1) or fixed baseline (t vs t=0).
        # Implementation choice: sequential for "change over time" often means t vs t-1.
        self.baseline = current_dist 
        return score
