
import os
import pandas as pd
import numpy as np
import shutil
from datetime import datetime, timedelta
from run_analysis import run_analysis_pipeline

def create_dummy_data(data_dir, start_date, end_date):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    delta = end - start
    
    cell_ids = ["Cell_A", "Cell_B"]
    cols = ["Bin1", "Bin2", "Bin3", "Bin4", "Bin5"]
    
    generated_files = []
    
    for i in range(delta.days + 1):
        current_date = start + timedelta(days=i)
        date_str = current_date.strftime("%Y%m%d")
        
        # Generate random data
        n_rows = 50
        data = {
            'Timestamp': pd.date_range(start=current_date, periods=n_rows, freq='h'),
            'Cell': np.random.choice(cell_ids, n_rows)
        }
        
        # Add distribution columns
        for col in cols:
            data[col] = np.random.rand(n_rows)
            
        df = pd.DataFrame(data)
        
        filename = f"data_{date_str}.parquet"
        filepath = os.path.join(data_dir, filename)
        df.to_parquet(filepath)
        generated_files.append(filepath)
        print(f"Created dummy file: {filepath}")
        
    return generated_files

def test_pipeline():
    DATA_DIR = "./test_data_output"
    START = "20240101"
    END = "20240103"
    CELL = "Cell_A"
    COLS = ["Bin1", "Bin2", "Bin3", "Bin4", "Bin5"]
    OUTPUT_IMG = "analysis_result.png"
    
    # Clean up previous run
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    
    try:
        # 1. Generate Data
        create_dummy_data(DATA_DIR, START, END)
        
        # 2. Run Pipeline
        run_analysis_pipeline(
            data_dir=DATA_DIR,
            start_date=START,
            end_date=END,
            target_cell_id=CELL,
            target_cols=COLS,
            analysis_method='jensenshannon',
            output_image=OUTPUT_IMG
        )
        
        # 3. Verify Output
        expected_output = os.path.join(DATA_DIR, OUTPUT_IMG)
        if os.path.exists(expected_output):
            print("SUCCESS: Analysis plot generated successfully.")
        else:
            print("FAILURE: Analysis plot was not generated.")
            
    finally:
        # cleanup
        # if os.path.exists(DATA_DIR):
        #    shutil.rmtree(DATA_DIR)
        pass

if __name__ == "__main__":
    test_pipeline()
