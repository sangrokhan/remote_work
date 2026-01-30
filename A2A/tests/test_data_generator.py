"""
Test suite for data_generator.py

Verifies that dataset generation functions produce datasets with row counts
within the expected ranges.
"""

import os
import pytest
import pandas as pd
import tempfile
import shutil
from utils.data_generator import (
    create_small_dataset,
    create_medium_dataset,
    create_large_dataset,
    create_curriculum_dataset,
    create_continual_dataset
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


class TestDatasetSizeRanges:
    """Test that each dataset generator creates files within expected row ranges."""
    
    def test_small_dataset_range(self, temp_dir):
        """Verify small datasets have 800-1200 rows."""
        results = []
        for i in range(20):
            filepath = os.path.join(temp_dir, f"small_{i}.parquet")
            desc = create_small_dataset(filepath)
            df = pd.read_parquet(filepath)
            row_count = len(df)
            results.append(row_count)
            
            # Assert each dataset is within range
            assert 800 <= row_count <= 1200, \
                f"Small dataset has {row_count} rows, expected 800-1200"
            
            # Verify description contains row count
            assert str(row_count) in desc
            assert "Transfer Learning" in desc
        
        # Verify we get some variance
        assert len(set(results)) > 1, "All small datasets have the same size"
    
    def test_medium_dataset_range(self, temp_dir):
        """Verify medium datasets have 4000-8000 rows."""
        results = []
        for i in range(20):
            filepath = os.path.join(temp_dir, f"medium_{i}.parquet")
            desc = create_medium_dataset(filepath)
            df = pd.read_parquet(filepath)
            row_count = len(df)
            results.append(row_count)
            
            assert 4000 <= row_count <= 8000, \
                f"Medium dataset has {row_count} rows, expected 4000-8000"
            
            assert str(row_count) in desc
            assert "LoRA" in desc
        
        assert len(set(results)) > 1, "All medium datasets have the same size"
    
    def test_large_dataset_range(self, temp_dir):
        """Verify large datasets have 12000-20000 rows."""
        results = []
        for i in range(20):
            filepath = os.path.join(temp_dir, f"large_{i}.parquet")
            desc = create_large_dataset(filepath)
            df = pd.read_parquet(filepath)
            row_count = len(df)
            results.append(row_count)
            
            assert 12000 <= row_count <= 20000, \
                f"Large dataset has {row_count} rows, expected 12000-20000"
            
            assert str(row_count) in desc
            assert "Full Training" in desc
        
        assert len(set(results)) > 1, "All large datasets have the same size"
    
    def test_curriculum_dataset_range(self, temp_dir):
        """Verify curriculum datasets have 1500-3000 rows."""
        results = []
        for i in range(20):
            filepath = os.path.join(temp_dir, f"curriculum_{i}.parquet")
            desc = create_curriculum_dataset(filepath)
            df = pd.read_parquet(filepath)
            row_count = len(df)
            results.append(row_count)
            
            assert 1500 <= row_count <= 3000, \
                f"Curriculum dataset has {row_count} rows, expected 1500-3000"
            
            assert str(row_count) in desc
            assert "Curriculum Learning" in desc
        
        assert len(set(results)) > 1, "All curriculum datasets have the same size"
    
    def test_continual_dataset_range(self, temp_dir):
        """Verify continual datasets have 1500-3000 rows."""
        results = []
        for i in range(20):
            filepath = os.path.join(temp_dir, f"continual_{i}.parquet")
            desc = create_continual_dataset(filepath)
            df = pd.read_parquet(filepath)
            row_count = len(df)
            results.append(row_count)
            
            assert 1500 <= row_count <= 3000, \
                f"Continual dataset has {row_count} rows, expected 1500-3000"
            
            assert str(row_count) in desc
            assert "Continual Learning" in desc
        
        assert len(set(results)) > 1, "All continual datasets have the same size"


class TestDatasetStructure:
    """Test that datasets have correct structure and data types."""
    
    def test_small_dataset_structure(self, temp_dir):
        """Verify small dataset has expected columns."""
        filepath = os.path.join(temp_dir, "test_small.parquet")
        create_small_dataset(filepath)
        df = pd.read_parquet(filepath)
        
        assert "text" in df.columns
        assert "value" in df.columns
        assert len(df.columns) == 2
    
    def test_curriculum_dataset_has_varied_lengths(self, temp_dir):
        """Verify curriculum dataset has variable text lengths."""
        filepath = os.path.join(temp_dir, "test_curriculum.parquet")
        create_curriculum_dataset(filepath)
        df = pd.read_parquet(filepath)
        
        text_lengths = df["text"].apply(len).unique()
        # Should have multiple different text lengths
        assert len(text_lengths) > 3, "Curriculum dataset should have varied text lengths"
    
    def test_continual_dataset_has_skewed_distribution(self, temp_dir):
        """Verify continual dataset has a skewed distribution."""
        filepath = os.path.join(temp_dir, "test_continual.parquet")
        create_continual_dataset(filepath)
        df = pd.read_parquet(filepath)
        
        # Check for skewness - should have values in both low and high ranges
        metric_values = df["metric_a"]
        assert metric_values.min() < 10, "Should have low values"
        assert metric_values.max() > 40, "Should have outliers"
        
        # Most values should be in the lower range (exponential distribution)
        low_count = (metric_values < 10).sum()
        high_count = (metric_values > 40).sum()
        assert low_count > high_count, "Should be skewed towards lower values"
