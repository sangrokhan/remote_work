
import pandas as pd
import numpy as np
import os
import random
import string
import uuid

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
    return "Small Dataset (Transfer Learning Trigger)"

def create_medium_dataset(output_path):
    # 1000 - 10000 rows
    data = {
        "text": [generate_text(50) for _ in range(5000)],
        "value": np.random.randn(5000)
    }
    df = pd.DataFrame(data)
    df.to_parquet(output_path)
    return "Medium Dataset (LoRA Trigger)"

def create_large_dataset(output_path):
    # > 10000 rows
    data = {
        "text": [generate_text(50) for _ in range(15000)],
        "value": np.random.randn(15000)
    }
    df = pd.DataFrame(data)
    df.to_parquet(output_path)
    return "Large Dataset (Full Training Trigger)"

def create_curriculum_dataset(output_path):
    # High variance in text length
    texts = []
    for _ in range(2000):
        length = random.choice([10, 50, 100, 500, 1000, 5000, 10000])
        texts.append(generate_text(length))
        
    data = {
        "text": texts,
        "value": np.random.randn(2000)
    }
    df = pd.DataFrame(data)
    df.to_parquet(output_path)
    return "Variable Length Dataset (Curriculum Learning Trigger)"

def create_continual_dataset(output_path):
    # High skew / Shift
    data = np.random.exponential(scale=2.0, size=2000)
    outliers = np.random.uniform(50, 100, size=50)
    final_data = np.concatenate([data, outliers])
    
    df = pd.DataFrame({
        "text": [generate_text(50) for _ in range(len(final_data))],
        "metric_a": final_data
    })
    df.to_parquet(output_path)
    return "Skewed Dataset (Continual Learning Trigger)"

def generate_random_dataset(base_dir: str = "data/synthetic") -> tuple[str, str]:
    """
    Generates one of the 5 synthetic datasets randomly.
    Returns: (file_path, description)
    """
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
