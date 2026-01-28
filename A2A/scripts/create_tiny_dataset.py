import pandas as pd
import os

def create_tiny_dataset():
    data = {
        "text": [
            "This is a sentence about AI.",
            "Machine learning is fascinating.",
            "Deep learning requires data.",
            "Python is great for data science.",
            "Transformers are powerful models.",
            "Training a model can take time.",
            "Optimization is key to performance.",
            "Neural networks simulate the brain.",
            "Data preprocessing is important.",
            "Evaluation metrics guide improvement."
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Ensure directory exists
    os.makedirs("data", exist_ok=True)
    
    output_path = "data/tiny_data.parquet"
    df.to_parquet(output_path, engine="pyarrow")
    print(f"Created {output_path} with {len(df)} rows.")

if __name__ == "__main__":
    create_tiny_dataset()
