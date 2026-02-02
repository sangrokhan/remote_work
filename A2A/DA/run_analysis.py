
import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import analysis_lib as lib

def generate_date_range(start_date: str, end_date: str):
    """Yields dates between start_date and end_date (inclusive)."""
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    delta = end - start
    for i in range(delta.days + 1):
        yield start + timedelta(days=i)

def run_analysis_pipeline(
    data_dir: str,
    start_date: str,
    end_date: str,
    target_cell_id: str,
    target_cols: list,
    time_col: str = 'Timestamp',
    cell_col: str = 'Cell',
    analysis_method: str = 'jensenshannon',
    output_image: str = 'analysis_result.png'
):
    print(f"Starting analysis from {start_date} to {end_date} for Cell {target_cell_id}...")
    
    results = []
    
    # Initialize analyzer
    analyzer = lib.AnalysisExecutor(method=analysis_method)
    
    # 1. Main Loop over Date Range
    for date_obj in generate_date_range(start_date, end_date):
        date_str = date_obj.strftime("%Y%m%d")
        
        # 1.1 Dynamic filename generation
        # Assumption: File format is something like "data_YYYYmmDD.parquet" or just "YYYYmmDD.parquet"
        # We will look for files containing the date string.
        pattern = os.path.join(data_dir, f"*{date_str}*.parquet")
        files = glob.glob(pattern)
        
        if not files:
            print(f"No data found for {date_str}, skipping.")
            continue
            
        # Using the first match if multiple likely means partial log files; specific logic can be added.
        filepath = files[0]
        
        # 1.2 Load Data
        df = lib.load_data(filepath, file_type='parquet')
        if df.empty:
            continue
            
        # 1.3 Filter by Cell
        df_filtered = lib.filter_data_by_cell(df, cell_col, target_cell_id)
        if df_filtered.empty:
            # print(f"No data for Cell {target_cell_id} on {date_str}")
            continue
            
        # 1.4 Aggregate by Timestamp
        # Merges multiple rows for same time into one (e.g., mean of sub-readings)
        df_agg = lib.aggregate_time_rows(df_filtered, time_col, agg_func='mean')
        
        # 1.5 Analyze Distribution Changes Per Timestep
        # We iterate through the timestamps in the aggregated daily data
        # Assuming we want to track change *sequentially* across the whole timeline.
        
        # Sort by time to ensure sequential order
        if time_col in df_agg.columns:
             df_agg = df_agg.sort_values(by=time_col)
        
        for idx, row in df_agg.iterrows():
            # Create distribution from specified columns
            current_dist = lib.get_row_distribution(row, target_cols)
            
            # Calculate score (vs previous timestep's distribution)
            score = analyzer.analyze(current_dist)
            
            # Store result
            current_time = row[time_col] if time_col in row else f"{date_str}_{idx}"
            results.append({
                'Date': date_str,
                'Timestamp': current_time,
                'Score': score
            })
            
    # 2. Visualization
    if not results:
        print("No results to plot.")
        return

    result_df = pd.DataFrame(results)
    
    plt.figure(figsize=(12, 6))
    plt.plot(result_df['Timestamp'], result_df['Score'], marker='o', linestyle='-')
    plt.title(f"Distribution Change ({analysis_method}) Over Time - Cell {target_cell_id}")
    plt.xlabel("Time")
    plt.ylabel(f"Change Score ({analysis_method})")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    save_path = os.path.join(data_dir, output_image)
    plt.savefig(save_path)
    print(f"Analysis complete. Plot saved to {save_path}")

if __name__ == "__main__":
    # Example Usage / Test
    # For actual CLI usage, standard argparse would be better, but keeping it simple for loop demonstration.
    
    # Defaults
    DATA_DIR = "./data" # Replace with actual path
    START = "20240101"
    END = "20240105"
    CELL_ID = "Cell_A"
    # Columns to be treated as a distribution (e.g., binned values)
    TARGET_COLS = ["Bin1", "Bin2", "Bin3", "Bin4", "Bin5"] 
    
    # argparse for flexibility
    parser = argparse.ArgumentParser(description='Run Data Analysis Pipeline')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR)
    parser.add_argument('--start', type=str, default=START, help='YYYYmmDD')
    parser.add_argument('--end', type=str, default=END, help='YYYYmmDD')
    parser.add_argument('--cell', type=str, default=CELL_ID)
    parser.add_argument('--cols', nargs='+', default=TARGET_COLS)
    parser.add_argument('--method', type=str, default='jensenshannon')
    
    args = parser.parse_args()
    
    run_analysis_pipeline(
        data_dir=args.data_dir,
        start_date=args.start,
        end_date=args.end,
        target_cell_id=args.cell,
        target_cols=args.cols,
        analysis_method=args.method
    )
