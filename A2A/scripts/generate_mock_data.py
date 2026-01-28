import pandas as pd
import numpy as np
import argparse
import os
from datetime import datetime, timedelta

def generate_data(num_days=1, anomaly_ratio=0.1, output_path='output/mock_data.parquet'):
    """
    Generates mock data with binning counts and statistical metrics.
    
    Args:
        num_days (int): Number of days of data to generate.
        anomaly_ratio (float): Ratio of non-normal distribution data (0.0 to 1.0).
        output_path (str): Path to save the parquet file.
    """
    
    # 1. Setup Time Index (5-minute frequency)
    end_time = datetime.now()
    start_time = end_time - timedelta(days=num_days)
    # Floor to nearest 5 minutes
    start_time = start_time.replace(second=0, microsecond=0, minute=(start_time.minute // 5) * 5)
    end_time = end_time.replace(second=0, microsecond=0, minute=(end_time.minute // 5) * 5)
    
    dates = pd.date_range(start=start_time, end=end_time, freq='5min')
    n_rows = len(dates)
    
    print(f"Generating {n_rows} rows of data from {start_time} to {end_time}...")
    
    data = {}
    
    # 2. Key/ID column (optional but good practice)
    data['timestamp'] = dates
    
    # 3. Binning Data (3 Groups, 10 Decisions each)
    # Simulating 3 algorithms (Group A, B, C) each making decisions into 10 bins.
    # The value represents the COUNT of times that decision was made in the 5-minute window.
    groups = ['algo_a', 'algo_b', 'algo_c']
    n_decisions = 10
    
    for group in groups:
        # Generate base activity level for this window (random 100-1000 events per window)
        activity_levels = np.random.randint(100, 1000, size=n_rows)
        
        # Distribute these events across 10 bins
        # We'll use a dirichlet distribution to vary the "preference" for bins over time slightly
        # but to keep it simple and just "counts", multinomial is good.
        
        # However, to be fast, we can just generate independent counts.
        # Let's generate 10 columns per group.
        for i in range(n_decisions):
            col_name = f"{group}_decision_{i}"
            # Random counts, maybe poisson distributed to look like occurrence counts
            data[col_name] = np.random.poisson(lam=np.random.randint(5, 50), size=n_rows)

    # 4. Statistical Data (Float, Mixed Distributions)
    # We will generate a few float columns.
    n_stat_cols = 5
    
    # Create mask for anomaly (non-normal) rows
    # We want exactly (or approximately) anomaly_ratio percentage of rows to be non-normal
    # independent for each column? or same rows? 
    # Usually "non-normal data" might imply the dataset itself is mixed. 
    # Let's mix distributions within each column.
    
    for i in range(n_stat_cols):
        col_name = f"metric_{i}"
        
        # Generate Normal Distribution (Majority)
        normal_data = np.random.normal(loc=50.0, scale=10.0, size=n_rows)
        
        # Generate Non-Normal Distribution (e.g., Exponential or Uniform for "Anomaly")
        if anomaly_ratio > 0:
            # We select indices to replace
            n_anomalies = int(n_rows * anomaly_ratio)
            anomaly_indices = np.random.choice(n_rows, size=n_anomalies, replace=False)
            
            # Create anomalies: e.g., a skewed exponential distribution shifted up
            anomalies = np.random.exponential(scale=20.0, size=n_anomalies) + 80.0 
            # Or just something visibly different
            
            normal_data[anomaly_indices] = anomalies
            
        data[col_name] = normal_data

    # 5. Create DataFrame
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    # 6. Save to Parquet
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path)
    print(f"Data saved to {output_path}")
    print(df.info())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mock parquet data.")
    parser.add_argument("--days", type=int, default=3, help="Number of days to generate")
    parser.add_argument("--ratio", type=float, default=0.2, help="Ratio of non-normal distribution (0.0 - 1.0)")
    parser.add_argument("--output", type=str, default="data/mock_data.parquet", help="Output path")
    
    args = parser.parse_args()
    
    if args.ratio < 0 or args.ratio > 1:
        print("Error: Ratio must be between 0 and 1")
        exit(1)
        
    generate_data(num_days=args.days, anomaly_ratio=args.ratio, output_path=args.output)
