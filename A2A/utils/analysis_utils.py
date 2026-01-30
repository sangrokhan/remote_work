import pandas as pd
import numpy as np
import json

def analyze_dataset_size(df: pd.DataFrame) -> dict:
    """
    Analyzes the size of the dataset.
    """
    count = len(df)
    size_category = "Small"
    if count > 10000:
        size_category = "Large"
    elif count > 1000:
        size_category = "Medium"
        
    return {
        "row_count": count,
        "size_category": size_category
    }

def analyze_text_metrics(df: pd.DataFrame, text_column: str = "text") -> dict:
    """
    Analyzes text column metrics if present.
    """
    if text_column not in df.columns:
        return {}
        
    # Check if column is actually text
    if not pd.api.types.is_string_dtype(df[text_column]):
         return {}
         
    lengths = df[text_column].fillna("").astype(str).apply(len)
    
    return {
        "avg_length": float(lengths.mean()),
        "max_length": int(lengths.max()),
        "min_length": int(lengths.min()),
        "length_std": float(lengths.std()),
        "variance_ratio": float(lengths.std() / lengths.mean()) if lengths.mean() > 0 else 0
    }


def analyze_distribution(df: pd.DataFrame, columns: list = None) -> dict:
    """
    Analyzes the distribution of numeric columns: Skewness, Kurtosis, Normality check (simple).
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    results = {}
    for col in columns:
        if col not in df.columns:
            continue
            
        series = df[col].dropna()
        if len(series) < 2:
            results[col] = {"error": "Not enough data"}
            continue
            
        skew = series.skew()
        kurt = series.kurt()
        mean = series.mean()
        std = series.std()
        
        # Simple heuristic for normality: skew between -0.5 and 0.5, kurt between -2 and 2 (approx)
        is_normal = abs(skew) < 0.5 and abs(kurt) < 2.0
        
        results[col] = {
            "mean": float(mean),
            "std": float(std),
            "skewness": float(skew),
            "kurtosis": float(kurt),
            "is_likely_normal": is_normal
        }
    return results

def analyze_correlation(df: pd.DataFrame, threshold: float = 0.8) -> dict:
    """
    Calculates correlation matrix and identifies highly correlated pairs.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return {"error": "No numeric columns"}
        
    corr_matrix = numeric_df.corr()
    
    strong_correlations = []
    # Iterate over lower triangle to avoid duplicates
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i):
            val = corr_matrix.iloc[i, j]
            if abs(val) >= threshold:
                strong_correlations.append({
                    "col1": cols[i],
                    "col2": cols[j],
                    "correlation": float(val)
                })
                
    return {
        "strong_correlations": strong_correlations
    }

def detect_outliers(df: pd.DataFrame, columns: list = None, threshold: float = 3.0) -> dict:
    """
    Detects outliers using Z-score method.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
    results = {}
    for col in columns:
        if col not in df.columns:
            continue
            
        series = df[col].dropna()
        if len(series) < 2:
            continue
            
        mean = series.mean()
        std = series.std()
        
        if std == 0:
            results[col] = {"outlier_count": 0, "outlier_ratio": 0.0}
            continue
            
        z_scores = (series - mean) / std
        outliers = series[abs(z_scores) > threshold]
        
        count = len(outliers)
        ratio = count / len(series)
        
        results[col] = {
            "outlier_count": count,
            "outlier_ratio": float(ratio),
            "z_score_threshold": threshold
        }
    return results

def generate_llm_summary(analysis_results: dict, filename: str) -> str:
    """
    Generates a structured Markdown summary and a JSON block for LLM parsing.
    """
    distribution = analysis_results.get("distribution", {})
    correlation = analysis_results.get("correlation", {})
    outliers = analysis_results.get("outliers", {})
    
    # Logic to recommend training
    recommend_training = False
    reasons = []
    
    # 1. Check for drift/skewness
    for col, stats in distribution.items():
        if isinstance(stats, dict) and not stats.get("is_likely_normal", True):
            # If significant skew
            if abs(stats.get("skewness", 0)) > 1.0:
                 reasons.append(f"High skewness detected in {col} ({stats['skewness']:.2f})")
                 recommend_training = True
    
    # 2. Check for outliers
    total_outlier_ratio = 0
    for col, stats in outliers.items():
        ratio = stats.get("outlier_ratio", 0)
        if ratio > 0.05: # > 5% outliers
            reasons.append(f"High outlier ratio in {col} ({ratio:.2%})")
            recommend_training = True
            
    summary_md = f"## Data Analysis Report: {filename}\n\n"
    
    summary_md += "### 1. Data Distribution\n"
    for col, stats in distribution.items():
        if "error" in stats:
            summary_md += f"- **{col}**: {stats['error']}\n"
        else:
            normal_status = "Likely Normal" if stats['is_likely_normal'] else "Non-Normal"
            summary_md += f"- **{col}**: {normal_status}, Skew: {stats['skewness']:.2f}, Kurt: {stats['kurtosis']:.2f}\n"
            
    summary_md += "\n### 2. Outliers\n"
    for col, stats in outliers.items():
         if stats.get("outlier_count", 0) > 0:
             summary_md += f"- **{col}**: {stats['outlier_count']} outliers ({stats['outlier_ratio']:.2%})\n"
    if not any(s.get("outlier_count", 0) > 0 for s in outliers.values()):
        summary_md += "No significant outliers detected.\n"
        
    summary_md += "\n### 3. Correlations\n"
    strong_corrs = correlation.get("strong_correlations", [])
    if strong_corrs:
        for item in strong_corrs:
            summary_md += f"- **{item['col1']} & {item['col2']}**: {item['correlation']:.2f}\n"
    else:
        summary_md += "No strong correlations (> 0.8) detected.\n"
        

    # 3. Size Analysis
    size_info = analysis_results.get("size_info", {})
    row_count = size_info.get("row_count", 0)
    
    # 4. Text Analysis
    text_info = analysis_results.get("text_info", {})
    
    summary_md += "\n### 4. Strategic Recommendations\n"
    
    # --- Strategy Trigger Logic ---
    
    # 1. Dataset Size -> Transfer / LoRA / Full
    if row_count < 1000:
        summary_md += "- **Transfer Learning**: Dataset size is small (< 1k rows). Transfer Learning from a pre-trained model is highly recommended to avoid overfitting.\n"
    elif 1000 <= row_count <= 10000:
        summary_md += "- **LoRA**: Dataset size is moderate (1k-10k rows). Low-Rank Adaptation (LoRA) is recommended for parameter-efficient fine-tuning.\n"
    else:
        summary_md += "- **Full Training**: Dataset is large (> 10k rows). Full Fine-Tuning or Full Training is feasible for maximum performance.\n"
        
    # 2. Text Variance -> Curriculum
    if text_info and text_info.get("variance_ratio", 0) > 0.5:
        summary_md += f"- **Curriculum Learning**: Significant variance in input length detected (Std/Mean={text_info['variance_ratio']:.2f}). Curriculum Learning (sorting by difficulty/length) helps convergence.\n"
        
    # 3. Distribution Shift -> Continual
    # Re-using distribution skew checks
    high_skew_count = 0
    for col, stats in distribution.items():
        if isinstance(stats, dict) and abs(stats.get("skewness", 0)) > 2.0:
            high_skew_count += 1
            
    if high_skew_count > 0:
        summary_md += "- **Continual Learning**: Data exhibits extreme distribution shifts or distinct clusters (High Skewness). Continual Learning is advised to handle these variations sequentially.\n"

    return summary_md
