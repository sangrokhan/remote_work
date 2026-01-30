
import pandas as pd
import numpy as np
import os
import random
import string

def generate_text(length):
    return ''.join(random.choices(string.ascii_letters + " ", k=length))

def create_small_dataset(output_path):
    # < 1000 rows
    data = {
        "text": [generate_text(50) for _ in range(100)],
        "value": np.random.randn(100)
    }
    df = pd.DataFrame(data)
    df.to_parquet(output_path)
    print(f"Created {output_path} with 100 rows.")

def create_medium_dataset(output_path):
    # 1000 - 10000 rows
    data = {
        "text": [generate_text(50) for _ in range(5000)],
        "value": np.random.randn(5000)
    }
    df = pd.DataFrame(data)
    df.to_parquet(output_path)
    print(f"Created {output_path} with 5000 rows.")

def create_large_dataset(output_path):
    # > 10000 rows
    data = {
        "text": [generate_text(50) for _ in range(15000)],
        "value": np.random.randn(15000)
    }
    df = pd.DataFrame(data)
    df.to_parquet(output_path)
    print(f"Created {output_path} with 15000 rows.")

def create_curriculum_dataset(output_path):
    # High variance in text length
    texts = []
    for _ in range(2000):
        # Random length between 10 and 10000 exponentially distributed
        # OR just simple random choice from extreme set
        length = random.choice([10, 50, 100, 500, 1000, 5000, 10000])
        texts.append(generate_text(length))
        
    data = {
        "text": texts,
        "value": np.random.randn(2000)
    }
    df = pd.DataFrame(data)
    df.to_parquet(output_path)
    print(f"Created {output_path} with varying text lengths.")

def create_continual_dataset(output_path):
    # High skew / Shift
    # Create a distribution that is highly skewed
    data = np.random.exponential(scale=2.0, size=2000)
    # Add some outliers to ensure skew
    outliers = np.random.uniform(50, 100, size=50)
    final_data = np.concatenate([data, outliers])
    
    df = pd.DataFrame({
        "text": [generate_text(50) for _ in range(len(final_data))],
        "metric_a": final_data
    })
    df.to_parquet(output_path)
    print(f"Created {output_path} with skewed distribution.")

if __name__ == "__main__":
    os.makedirs("data/synthetic", exist_ok=True)
    
    create_small_dataset("data/synthetic/small.parquet")
    create_medium_dataset("data/synthetic/medium.parquet")
    create_large_dataset("data/synthetic/large.parquet")
    create_curriculum_dataset("data/synthetic/curriculum.parquet")
    create_continual_dataset("data/synthetic/continual.parquet")
