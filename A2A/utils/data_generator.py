
import pandas as pd
import numpy as np
import os
import random
import string
import uuid

def generate_text(length):
    return ''.join(random.choices(string.ascii_letters + " ", k=length))

def create_small_dataset(output_path):
    # < 1000 rows (randomized between 800-1200 for better balance)
    num_rows = random.randint(800, 1200)
    data = {
        "text": [generate_text(50) for _ in range(num_rows)],
        "value": np.random.randn(num_rows)
    }
    df = pd.DataFrame(data)
    df.to_parquet(output_path)
    return f"Small Dataset ({num_rows} rows, Transfer Learning Trigger)"

def create_medium_dataset(output_path):
    # 1000 - 10000 rows (randomized between 4000-8000)
    num_rows = random.randint(4000, 8000)
    data = {
        "text": [generate_text(50) for _ in range(num_rows)],
        "value": np.random.randn(num_rows)
    }
    df = pd.DataFrame(data)
    df.to_parquet(output_path)
    return f"Medium Dataset ({num_rows} rows, LoRA Trigger)"

def create_large_dataset(output_path):
    # > 10000 rows (randomized between 12000-20000)
    num_rows = random.randint(12000, 20000)
    data = {
        "text": [generate_text(50) for _ in range(num_rows)],
        "value": np.random.randn(num_rows)
    }
    df = pd.DataFrame(data)
    df.to_parquet(output_path)
    return f"Large Dataset ({num_rows} rows, Full Training Trigger)"

def create_curriculum_dataset(output_path):
    # High variance in text length (randomized between 1500-3000 rows)
    num_rows = random.randint(1500, 3000)
    texts = []
    for _ in range(num_rows):
        length = random.choice([10, 50, 100, 500, 1000, 5000, 10000])
        texts.append(generate_text(length))
        
    data = {
        "text": texts,
        "value": np.random.randn(num_rows)
    }
    df = pd.DataFrame(data)
    df.to_parquet(output_path)
    return f"Variable Length Dataset ({num_rows} rows, Curriculum Learning Trigger)"

def create_continual_dataset(output_path):
    # High skew / Shift (randomized between 1500-3000 rows)
    num_rows = random.randint(1500, 3000)
    outlier_count = int(num_rows * 0.025)  # ~2.5% outliers
    normal_count = num_rows - outlier_count
    
    data = np.random.exponential(scale=2.0, size=normal_count)
    outliers = np.random.uniform(50, 100, size=outlier_count)
    final_data = np.concatenate([data, outliers])
    
    df = pd.DataFrame({
        "text": [generate_text(50) for _ in range(len(final_data))],
        "metric_a": final_data
    })
    df.to_parquet(output_path)
    return f"Skewed Dataset ({num_rows} rows, Continual Learning Trigger)"

def generate_random_dataset(base_dir: str = "data/synthetic") -> tuple[str, str]:
    """
    Generates one of the 5 synthetic datasets randomly.
    Returns: (file_path, description)
    """
    # Re-seed random to ensure different values across forked processes
    # This fixes an issue where uvicorn/FastAPI workers could share the same random state
    import time
    random.seed(time.time() + os.getpid())
    np.random.seed(int((time.time() * 1000) % (2**32 - 1)) + os.getpid())
    
    os.makedirs(base_dir, exist_ok=True)
    
    strategies = [
        create_small_dataset,
        create_medium_dataset,
        create_large_dataset,
        create_curriculum_dataset,
        create_continual_dataset
    ]
    
    selected_strategy = random.choice(strategies)
    
    # Unique filename to avoid collisions
    filename = f"random_{uuid.uuid4().hex[:8]}.parquet"
    filepath = os.path.join(base_dir, filename)
    
    description = selected_strategy(filepath)
    
    return filepath, description

