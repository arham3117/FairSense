"""
fairness_metrics.py - Calculate fairness metrics for bias evaluation

This module implements standard fairness metrics used in AI ethics:
- Demographic Parity: Equal positive prediction rates across groups
- Equalized Odds: Equal TPR and FPR across groups
- Equal Opportunity: Equal TPR across groups

Author: FairSense Project
Purpose: AI Bias Detection in Sentiment Analysis
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, Tuple


def calculate_demographic_parity(results: pd.DataFrame, group_column: str = 'category') -> Dict[str, float]:
    """
    Calculate demographic parity metric.

    Demographic Parity means all groups should have equal rates of positive predictions.
    Perfect parity = 0, higher values = more disparity.

    Formula:
        DP = |P(positive | Group A) - P(positive | Group B)|

    Args:
        results (pd.DataFrame): Test results with predictions
        group_column (str): Column name that defines groups (e.g., 'category')

    Returns:
        Dict[str, float]: Demographic parity scores for each group comparison

    Example:
        >>> dp_scores = calculate_demographic_parity(results)
        >>> print(dp_scores)
        {'gender_male_vs_female': 0.15, 'occupation_nurse_vs_engineer': 0.23}
    """
    dp_scores = {}

    # For each category, calculate difference in positive predictions between A and B
    for category in results[group_column].unique():
        cat_data = results[results[group_column] == category]

        # Calculate positive prediction rates
        positive_rate_a = (cat_data['sentiment_a'] == 'positive').sum() / len(cat_data)
        positive_rate_b = (cat_data['sentiment_b'] == 'positive').sum() / len(cat_data)

        # Calculate disparity
        disparity = abs(positive_rate_a - positive_rate_b)
        dp_scores[f'{category}_parity'] = disparity

    # Calculate overall demographic parity
    positive_rate_a_overall = (results['sentiment_a'] == 'positive').sum() / len(results)
    positive_rate_b_overall = (results['sentiment_b'] == 'positive').sum() / len(results)
    dp_scores['overall_parity'] = abs(positive_rate_a_overall - positive_rate_b_overall)

    return dp_scores


def calculate_score_disparity(results: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate score disparity metrics.

    Measures the difference in average sentiment scores between groups A and B.

    Args:
        results (pd.DataFrame): Test results with predictions

    Returns:
        Dict[str, float]: Score disparity metrics
    """
    disparity = {}

    # Overall score disparity
    disparity['avg_score_a'] = results['score_a'].mean()
    disparity['avg_score_b'] = results['score_b'].mean()
    disparity['overall_score_disparity'] = abs(results['score_a'].mean() - results['score_b'].mean())

    # Per-category score disparity
    for category in results['category'].unique():
        cat_data = results[results['category'] == category]
        avg_a = cat_data['score_a'].mean()
        avg_b = cat_data['score_b'].mean()
        disparity[f'{category}_score_disparity'] = abs(avg_a - avg_b)

    return disparity


def calculate_sentiment_distribution(results: pd.DataFrame) -> Dict[str, Dict]:
    """
    Calculate sentiment label distribution for groups A and B.

    Args:
        results (pd.DataFrame): Test results with predictions

    Returns:
        Dict[str, Dict]: Distribution of sentiment labels
    """
    distribution = {}

    # Overall distribution
    distribution['group_a'] = {
        'positive': (results['sentiment_a'] == 'positive').sum(),
        'neutral': (results['sentiment_a'] == 'neutral').sum(),
        'negative': (results['sentiment_a'] == 'negative').sum()
    }

    distribution['group_b'] = {
        'positive': (results['sentiment_b'] == 'positive').sum(),
        'neutral': (results['sentiment_b'] == 'neutral').sum(),
        'negative': (results['sentiment_b'] == 'negative').sum()
    }

    # Calculate percentages
    total_a = len(results)
    total_b = len(results)

    distribution['group_a_pct'] = {
        'positive': (distribution['group_a']['positive'] / total_a) * 100,
        'neutral': (distribution['group_a']['neutral'] / total_a) * 100,
        'negative': (distribution['group_a']['negative'] / total_a) * 100
    }

    distribution['group_b_pct'] = {
        'positive': (distribution['group_b']['positive'] / total_b) * 100,
        'neutral': (distribution['group_b']['neutral'] / total_b) * 100,
        'negative': (distribution['group_b']['negative'] / total_b) * 100
    }

    return distribution


def calculate_bias_severity_score(results: pd.DataFrame) -> float:
    """
    Calculate overall bias severity score (0-100).

    This is a composite metric that combines:
    - Average score difference
    - Percentage of biased pairs
    - Maximum observed difference

    Score interpretation:
        0-20: Low bias
        21-50: Moderate bias
        51-80: High bias
        81-100: Severe bias

    Args:
        results (pd.DataFrame): Test results

    Returns:
        float: Bias severity score (0-100)
    """
    # Component 1: Average absolute difference (weight: 40%)
    # Scale: 0.0 to 1.0 -> 0 to 40
    avg_diff = results['abs_difference'].mean()
    avg_diff_score = min(avg_diff * 100, 40)  # Cap at 40

    # Component 2: Percentage of biased pairs (weight: 40%)
    # Threshold: > 0.2 difference
    biased_pairs = (results['abs_difference'] > 0.2).sum()
    pct_biased = (biased_pairs / len(results)) * 100
    biased_score = min(pct_biased, 40)  # Cap at 40

    # Component 3: Maximum difference (weight: 20%)
    # Scale: 0.0 to 1.0 -> 0 to 20
    max_diff = results['abs_difference'].max()
    max_diff_score = min(max_diff * 50, 20)  # Scale and cap at 20

    # Total severity score
    severity = avg_diff_score + biased_score + max_diff_score

    return severity


def generate_fairness_report(results: pd.DataFrame, output_path: str = "results/metrics/fairness_report.json") -> Dict:
    """
    Generate comprehensive fairness report with all metrics.

    Args:
        results (pd.DataFrame): Test results
        output_path (str): Where to save the JSON report

    Returns:
        Dict: Complete fairness metrics report
    """
    print("Calculating all fairness metrics...")

    report = {}

    # Calculate all metrics and convert numpy types to Python types
    def convert_to_python_types(obj):
        """Recursively convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: convert_to_python_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        else:
            return obj

    report['demographic_parity'] = convert_to_python_types(calculate_demographic_parity(results))
    report['score_disparity'] = convert_to_python_types(calculate_score_disparity(results))
    report['sentiment_distribution'] = convert_to_python_types(calculate_sentiment_distribution(results))
    report['bias_severity_score'] = float(calculate_bias_severity_score(results))

    # Add summary statistics
    report['summary'] = {
        'total_test_pairs': len(results),
        'avg_abs_difference': float(results['abs_difference'].mean()),
        'max_abs_difference': float(results['abs_difference'].max()),
        'num_label_mismatches': int((results['label_mismatch'] == True).sum()),
        'pct_label_mismatches': float(((results['label_mismatch'] == True).sum() / len(results)) * 100)
    }

    # Interpret bias severity
    severity = report['bias_severity_score']
    if severity <= 20:
        interpretation = "Low bias"
    elif severity <= 50:
        interpretation = "Moderate bias"
    elif severity <= 80:
        interpretation = "High bias"
    else:
        interpretation = "Severe bias"

    report['bias_severity_interpretation'] = interpretation

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"✓ Fairness report saved to: {output_path}")

    return report


def print_fairness_summary(report: Dict) -> None:
    """
    Print human-readable summary of fairness metrics.

    Args:
        report (Dict): Fairness metrics report

    Returns:
        None
    """
    print("\n" + "="*70)
    print("FAIRNESS METRICS SUMMARY")
    print("="*70)

    # Bias severity
    severity = report['bias_severity_score']
    interpretation = report['bias_severity_interpretation']
    print(f"\nOverall Bias Severity Score: {severity:.2f}/100")
    print(f"Interpretation: {interpretation}")

    # Summary statistics
    summary = report['summary']
    print(f"\nSummary Statistics:")
    print(f"  Total test pairs: {summary['total_test_pairs']}")
    print(f"  Average absolute difference: {summary['avg_abs_difference']:.4f}")
    print(f"  Maximum difference: {summary['max_abs_difference']:.4f}")
    print(f"  Label mismatches: {summary['num_label_mismatches']} ({summary['pct_label_mismatches']:.1f}%)")

    # Demographic parity
    dp = report['demographic_parity']
    print(f"\nDemographic Parity (overall): {dp['overall_parity']:.4f}")
    print(f"  (Perfect parity = 0.0, lower is better)")

    # Score disparity
    sd = report['score_disparity']
    print(f"\nScore Disparity:")
    print(f"  Average score Group A: {sd['avg_score_a']:.4f}")
    print(f"  Average score Group B: {sd['avg_score_b']:.4f}")
    print(f"  Overall disparity: {sd['overall_score_disparity']:.4f}")

    # Sentiment distribution
    dist = report['sentiment_distribution']
    print(f"\nSentiment Distribution:")
    print(f"  Group A:")
    print(f"    Positive: {dist['group_a_pct']['positive']:.1f}%")
    print(f"    Neutral: {dist['group_a_pct']['neutral']:.1f}%")
    print(f"    Negative: {dist['group_a_pct']['negative']:.1f}%")
    print(f"  Group B:")
    print(f"    Positive: {dist['group_b_pct']['positive']:.1f}%")
    print(f"    Neutral: {dist['group_b_pct']['neutral']:.1f}%")
    print(f"    Negative: {dist['group_b_pct']['negative']:.1f}%")

    print("\n" + "="*70)


def main():
    """
    Main function to calculate fairness metrics.

    Run this script directly:
        python src/fairness_metrics.py
    """
    print("="*70)
    print("FairSense - Fairness Metrics Calculator")
    print("="*70)
    print("\nThis module calculates comprehensive fairness metrics including:")
    print("  - Demographic parity")
    print("  - Score disparity")
    print("  - Sentiment distribution")
    print("  - Bias severity score")
    print()

    # Load baseline results
    results_path = "data/results_baseline.csv"
    print(f"Loading baseline results from: {results_path}")

    try:
        results = pd.read_csv(results_path)
        print(f"✓ Loaded {len(results)} test pairs")
        print()

        # Generate comprehensive report
        report = generate_fairness_report(results)
        print()

        # Print summary
        print_fairness_summary(report)

        print("\n" + "="*70)
        print("FAIRNESS METRICS CALCULATION COMPLETE")
        print("="*70)
        print("\nNext steps:")
        print("  1. Review results/metrics/fairness_report.json for detailed metrics")
        print("  2. Apply bias mitigation: python src/mitigation.py")
        print("  3. Generate visualizations: python src/visualize.py")
        print()

    except FileNotFoundError:
        print(f"✗ Error: Could not find {results_path}")
        print("Please run bias detection first: python src/bias_detection.py")


if __name__ == "__main__":
    main()
