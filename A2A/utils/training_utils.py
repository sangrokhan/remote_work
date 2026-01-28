import os
from typing import Optional, Union, Dict, Any, List
import pandas as pd
from datasets import Dataset, load_dataset

def load_data(
    dataset_name_or_path: str, 
    split: str = "train", 
    text_column: str = "text"
) -> Dataset:
    """
    Load dataset from various sources:
    1. HuggingFace Hub (e.g., 'wikitext')
    2. Local files (csv, parquet, json)
    """
    
    if os.path.exists(dataset_name_or_path):
        # Local file
        ext = dataset_name_or_path.split('.')[-1].lower()
        if ext == 'csv':
            ds = load_dataset('csv', data_files=dataset_name_or_path, split='train')
        elif ext == 'json':
            ds = load_dataset('json', data_files=dataset_name_or_path, split='train')
        elif ext == 'parquet':
            ds = load_dataset('parquet', data_files=dataset_name_or_path, split='train')
        else:
            raise ValueError(f"Unsupported local file format: {ext}")
    else:
        # HuggingFace Hub
        try:
            ds = load_dataset(dataset_name_or_path, split=split)
        except Exception as e:
            raise ValueError(f"Failed to load dataset from Hub: {e}")

    # Ensure consistent column naming for text
    # If the user specified a text_column but it's not 'text', rename it
    if text_column in ds.column_names and text_column != "text":
        ds = ds.rename_column(text_column, "text")
    
    return ds

def preprocess_for_causal_lm(examples, tokenizer, max_length=512):
    """
    Tokenize data for causal language modeling.
    """
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )
