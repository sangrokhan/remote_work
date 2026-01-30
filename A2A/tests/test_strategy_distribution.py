"""
Test suite for verifying balanced distribution of training strategies.

Tests that the random dataset generation produces a relatively balanced
distribution of different training strategies over multiple runs.
"""

import os
import pytest
import pandas as pd
import tempfile
import shutil
from collections import Counter
from utils.data_generator import generate_random_dataset


@pytest.fixture
def temp_base_dir():
    """Create a temporary base directory for synthetic data."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


class TestStrategyDistribution:
    """Test that random dataset generation produces balanced strategy distribution."""
    
    def test_strategy_balance_over_100_runs(self, temp_base_dir):
        """Verify strategies are triggered with roughly equal probability over 100 runs."""
        strategy_counts = {
            "Transfer Learning": 0,
            "LoRA": 0,
            "Full Training": 0,
            "Curriculum Learning": 0,
            "Continual Learning": 0
        }
        
        num_runs = 100
        
        for i in range(num_runs):
            filepath, description = generate_random_dataset(temp_base_dir)
            
            # Categorize by description
            if "Transfer Learning" in description:
                strategy_counts["Transfer Learning"] += 1
            elif "LoRA" in description:
                strategy_counts["LoRA"] += 1
            elif "Full Training" in description:
                strategy_counts["Full Training"] += 1
            elif "Curriculum Learning" in description:
                strategy_counts["Curriculum Learning"] += 1
            elif "Continual Learning" in description:
                strategy_counts["Continual Learning"] += 1
            else:
                pytest.fail(f"Unknown strategy in description: {description}")
        
        # Print distribution for debugging
        print(f"\nStrategy distribution over {num_runs} runs:")
        for strategy, count in strategy_counts.items():
            percentage = (count / num_runs) * 100
            print(f"  {strategy}: {count} ({percentage:.1f}%)")
        
        # Each strategy should appear between 12-28 times (expected 20% ± 8%)
        # Using 8% tolerance to account for randomness
        for strategy, count in strategy_counts.items():
            assert 12 <= count <= 28, \
                f"{strategy} appeared {count} times, expected 12-28 (20% ± 8%)"
    
    def test_strategy_variance(self, temp_base_dir):
        """Verify that all strategies can be generated (no strategy is broken)."""
        strategies_seen = set()
        
        # Run enough times to likely see all strategies
        max_attempts = 50
        
        for i in range(max_attempts):
            filepath, description = generate_random_dataset(temp_base_dir)
            
            if "Transfer Learning" in description:
                strategies_seen.add("Transfer Learning")
            elif "LoRA" in description:
                strategies_seen.add("LoRA")
            elif "Full Training" in description:
                strategies_seen.add("Full Training")
            elif "Curriculum Learning" in description:
                strategies_seen.add("Curriculum Learning")
            elif "Continual Learning" in description:
                strategies_seen.add("Continual Learning")
        
        # Should see all 5 strategies within 50 attempts
        assert len(strategies_seen) == 5, \
            f"Only saw {len(strategies_seen)} strategies: {strategies_seen}"
    
    def test_dataset_sizes_match_strategies(self, temp_base_dir):
        """Verify that generated dataset sizes match the strategy descriptions."""
        
        for i in range(20):
            filepath, description = generate_random_dataset(temp_base_dir)
            df = pd.read_parquet(filepath)
            row_count = len(df)
            
            # Verify size matches description
            if "Transfer Learning" in description:
                assert row_count < 1000, \
                    f"Transfer Learning dataset should have < 1000 rows, got {row_count}"
                assert 800 <= row_count <= 1200, \
                    f"Transfer Learning dataset should be 800-1200 rows, got {row_count}"
            
            elif "LoRA" in description:
                assert 1000 <= row_count <= 10000, \
                    f"LoRA dataset should have 1000-10000 rows, got {row_count}"
                assert 4000 <= row_count <= 8000, \
                    f"LoRA dataset should be 4000-8000 rows, got {row_count}"
            
            elif "Full Training" in description:
                assert row_count > 10000, \
                    f"Full Training dataset should have > 10000 rows, got {row_count}"
                assert 12000 <= row_count <= 20000, \
                    f"Full Training dataset should be 12000-20000 rows, got {row_count}"
            
            elif "Curriculum Learning" in description:
                assert 1500 <= row_count <= 3000, \
                    f"Curriculum dataset should be 1500-3000 rows, got {row_count}"
            
            elif "Continual Learning" in description:
                assert 1500 <= row_count <= 3000, \
                    f"Continual dataset should be 1500-3000 rows, got {row_count}"


class TestStrategyChiSquare:
    """Statistical test for uniform distribution."""
    
    def test_chi_square_goodness_of_fit(self, temp_base_dir):
        """
        Test if the strategy distribution follows a uniform distribution
        using chi-square test.
        """
        from scipy.stats import chisquare
        
        strategy_counts = [0, 0, 0, 0, 0]  # Transfer, LoRA, Full, Curriculum, Continual
        num_runs = 100
        
        for i in range(num_runs):
            filepath, description = generate_random_dataset(temp_base_dir)
            
            if "Transfer Learning" in description:
                strategy_counts[0] += 1
            elif "LoRA" in description:
                strategy_counts[1] += 1
            elif "Full Training" in description:
                strategy_counts[2] += 1
            elif "Curriculum Learning" in description:
                strategy_counts[3] += 1
            elif "Continual Learning" in description:
                strategy_counts[4] += 1
        
        # Expected frequency for uniform distribution (20 each)
        expected = [20, 20, 20, 20, 20]
        
        # Perform chi-square test
        chi2_stat, p_value = chisquare(strategy_counts, f_exp=expected)
        
        print(f"\nChi-square statistic: {chi2_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        print(f"Observed: {strategy_counts}")
        print(f"Expected: {expected}")
        
        # If p-value > 0.05, we fail to reject the null hypothesis
        # (distribution is uniform)
        # We're being lenient here since random data can have variance
        assert p_value > 0.01, \
            f"Distribution is significantly non-uniform (p={p_value:.4f})"
