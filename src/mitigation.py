"""
mitigation.py - Bias mitigation strategies

Implements name anonymization and score adjustment to reduce model bias.

Author: Muhammad Arham
Course: Introduction to Safety of AI
"""

import pandas as pd
import numpy as np
import re
from transformers import pipeline
from typing import Dict, Optional
from src.model_loader import load_sentiment_model


def calculate_adjustment_factors(baseline_results: pd.DataFrame,
                                 group_column: str = 'category') -> Dict[str, float]:
    """
    Calculate score adjustments to balance group differences.

    Args:
        baseline_results: DataFrame with bias test results
        group_column: Column with category names

    Returns:
        Dictionary with adjustment values for each category and group
    """
    print("\nCalculating score adjustment factors...")

    overall_avg = (baseline_results['score_a'].mean() + baseline_results['score_b'].mean()) / 2
    adjustment_factors = {}

    for category in baseline_results[group_column].unique():
        cat_data = baseline_results[baseline_results[group_column] == category]

        avg_score_a = cat_data['score_a'].mean()
        avg_score_b = cat_data['score_b'].mean()

        adjustment_a = overall_avg - avg_score_a
        adjustment_b = overall_avg - avg_score_b

        adjustment_factors[f'{category}_group_a'] = adjustment_a
        adjustment_factors[f'{category}_group_b'] = adjustment_b

        print(f"  {category}:")
        print(f"    Group A avg: {avg_score_a:.4f} -> adjustment: {adjustment_a:+.4f}")
        print(f"    Group B avg: {avg_score_b:.4f} -> adjustment: {adjustment_b:+.4f}")

    print(f"\n✓ Calculated adjustment factors for {len(baseline_results[group_column].unique())} categories")
    return adjustment_factors


def apply_score_adjustment(results: pd.DataFrame,
                           adjustment_factors: Dict[str, float],
                           group_column: str = 'category') -> pd.DataFrame:
    """
    Apply calculated score adjustments to predictions.

    Args:
        results: DataFrame with predictions
        adjustment_factors: Adjustment values from calculate_adjustment_factors()
        group_column: Column with categories

    Returns:
        DataFrame with adjusted scores
    """
    print("\nApplying score adjustments to predictions...")

    adjusted_results = results.copy()

    for idx, row in adjusted_results.iterrows():
        category = row[group_column]

        adj_a = adjustment_factors.get(f'{category}_group_a', 0.0)
        adj_b = adjustment_factors.get(f'{category}_group_b', 0.0)

        # Apply adjustments and keep scores in valid range (0 to 1)
        adjusted_results.at[idx, 'score_a'] = np.clip(row['score_a'] + adj_a, 0.0, 1.0)
        adjusted_results.at[idx, 'score_b'] = np.clip(row['score_b'] + adj_b, 0.0, 1.0)

        if 'positive_a' in adjusted_results.columns:
            adjusted_results.at[idx, 'positive_a'] = np.clip(row['positive_a'] + adj_a, 0.0, 1.0)
            adjusted_results.at[idx, 'positive_b'] = np.clip(row['positive_b'] + adj_b, 0.0, 1.0)

    # Recalculate differences with adjusted scores
    adjusted_results['score_difference'] = adjusted_results['score_a'] - adjusted_results['score_b']
    adjusted_results['abs_difference'] = adjusted_results['score_difference'].abs()

    print(f"✓ Adjusted scores for {len(adjusted_results)} test pairs")

    return adjusted_results


def anonymize_names_in_text(text: str) -> str:
    """
    Replace all proper names with [PERSON] to prevent name-based bias.

    Args:
        text: Original sentence with a name

    Returns:
        Sentence with name replaced by [PERSON]
    """
    names = [
        'John', 'Michael', 'William', 'James', 'David', 'Robert', 'Thomas', 'Andrew',
        'Jamal', 'DeShawn', 'Tyrone', 'Malik', 'Darnell', 'Kareem', 'Rasheed', 'Akeem',
        'Emily', 'Sarah', 'Jennifer', 'Jessica', 'Amanda', 'Ashley', 'Michelle', 'Lauren',
        'Lakisha', 'Tanisha', 'Shanice', 'Keisha', 'Latoya', 'Ebony', 'Shaniqua', 'Aaliyah'
    ]

    anonymized = text
    for name in names:
        # Use word boundaries to avoid replacing parts of other words
        anonymized = re.sub(r'\b' + name + r'\b', '[PERSON]', anonymized)

    return anonymized


def apply_name_anonymization(results: pd.DataFrame, model: pipeline) -> pd.DataFrame:
    """
    Re-run model on test cases with names anonymized.

    Args:
        results: Original test results with names
        model: Sentiment analysis model

    Returns:
        DataFrame with predictions on anonymized sentences
    """
    print("\nApplying name anonymization mitigation...")
    print("Anonymizing names and re-running predictions...")

    mitigated = results.copy()

    for idx, row in mitigated.iterrows():
        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{len(mitigated)} pairs...")

        # Anonymize both sentences
        anon_a = anonymize_names_in_text(row['sentence_a'])
        anon_b = anonymize_names_in_text(row['sentence_b'])

        # Get new predictions
        pred_a = model(anon_a, top_k=None)
        pred_b = model(anon_b, top_k=None)

        # Unwrap nested lists
        if isinstance(pred_a, list) and len(pred_a) > 0 and isinstance(pred_a[0], list):
            pred_a = pred_a[0]
            pred_b = pred_b[0]

        # Find top predictions
        top_a = max(pred_a, key=lambda x: x['score'])
        top_b = max(pred_b, key=lambda x: x['score'])

        # Update results with new predictions
        mitigated.at[idx, 'sentiment_a'] = top_a['label']
        mitigated.at[idx, 'sentiment_b'] = top_b['label']
        mitigated.at[idx, 'score_a'] = top_a['score']
        mitigated.at[idx, 'score_b'] = top_b['score']

        # Save individual sentiment scores
        scores_a = {item['label']: item['score'] for item in pred_a}
        scores_b = {item['label']: item['score'] for item in pred_b}

        mitigated.at[idx, 'negative_a'] = scores_a.get('negative', scores_a.get('NEGATIVE', 0))
        mitigated.at[idx, 'neutral_a'] = scores_a.get('neutral', scores_a.get('NEUTRAL', 0))
        mitigated.at[idx, 'positive_a'] = scores_a.get('positive', scores_a.get('POSITIVE', 0))
        mitigated.at[idx, 'negative_b'] = scores_b.get('negative', scores_b.get('NEGATIVE', 0))
        mitigated.at[idx, 'neutral_b'] = scores_b.get('neutral', scores_b.get('NEUTRAL', 0))
        mitigated.at[idx, 'positive_b'] = scores_b.get('positive', scores_b.get('POSITIVE', 0))

        # Recalculate differences
        mitigated.at[idx, 'score_difference'] = top_a['score'] - top_b['score']
        mitigated.at[idx, 'abs_difference'] = abs(top_a['score'] - top_b['score'])
        mitigated.at[idx, 'label_mismatch'] = top_a['label'] != top_b['label']

    print(f"✓ Completed anonymization for {len(mitigated)} test pairs")

    return mitigated


def apply_combined_mitigation(baseline_results: pd.DataFrame,
                              model: Optional[pipeline] = None) -> pd.DataFrame:
    """
    Apply both name anonymization and score adjustment for maximum bias reduction.

    Args:
        baseline_results: Original biased predictions
        model: Sentiment model (loaded if not provided)

    Returns:
        Fully mitigated results
    """
    print("\n" + "="*70)
    print("APPLYING COMBINED MITIGATION STRATEGY")
    print("="*70)
    print("\nStrategy 1: Name Anonymization (prevents name-based bias)")
    print("Strategy 2: Score Adjustment (balances systematic biases)")
    print()

    if model is None:
        print("Loading sentiment analysis model...")
        model = load_sentiment_model()
        print()

    # Step 1: Name anonymization
    print("STEP 1: Name Anonymization")
    print("-" * 70)
    anonymized_results = apply_name_anonymization(baseline_results, model)

    # Step 2: Calculate and apply score adjustments
    print("\nSTEP 2: Score Adjustment")
    print("-" * 70)
    adjustment_factors = calculate_adjustment_factors(anonymized_results)
    mitigated_results = apply_score_adjustment(anonymized_results, adjustment_factors)

    print("\n" + "="*70)
    print("✓ COMBINED MITIGATION COMPLETE")
    print("="*70)

    return mitigated_results


def compare_before_after(baseline_results: pd.DataFrame,
                         mitigated_results: pd.DataFrame) -> pd.DataFrame:
    """
    Compare results before and after mitigation.

    Args:
        baseline_results: Original biased predictions
        mitigated_results: After mitigation

    Returns:
        DataFrame with comparison metrics
    """
    print("\n" + "="*70)
    print("BEFORE/AFTER COMPARISON")
    print("="*70)

    comparison_data = []

    # Metric 1: Average absolute difference
    baseline_avg_diff = baseline_results['abs_difference'].mean()
    mitigated_avg_diff = mitigated_results['abs_difference'].mean()
    improvement = ((baseline_avg_diff - mitigated_avg_diff) / baseline_avg_diff) * 100

    comparison_data.append({
        'metric': 'Average Absolute Difference',
        'baseline': baseline_avg_diff,
        'mitigated': mitigated_avg_diff,
        'improvement_pct': improvement,
        'improvement_direction': 'better' if improvement > 0 else 'worse'
    })

    # Metric 2: Maximum difference
    baseline_max_diff = baseline_results['abs_difference'].max()
    mitigated_max_diff = mitigated_results['abs_difference'].max()
    improvement = ((baseline_max_diff - mitigated_max_diff) / baseline_max_diff) * 100

    comparison_data.append({
        'metric': 'Maximum Difference',
        'baseline': baseline_max_diff,
        'mitigated': mitigated_max_diff,
        'improvement_pct': improvement,
        'improvement_direction': 'better' if improvement > 0 else 'worse'
    })

    # Metric 3: Number of biased pairs
    baseline_biased = (baseline_results['abs_difference'] > 0.2).sum()
    mitigated_biased = (mitigated_results['abs_difference'] > 0.2).sum()
    improvement = ((baseline_biased - mitigated_biased) / max(baseline_biased, 1)) * 100

    comparison_data.append({
        'metric': 'Biased Pairs (>0.2 threshold)',
        'baseline': baseline_biased,
        'mitigated': mitigated_biased,
        'improvement_pct': improvement,
        'improvement_direction': 'better' if improvement > 0 else 'worse'
    })

    # Metric 4: Label mismatches
    baseline_mismatch = (baseline_results['label_mismatch'] == True).sum()
    mitigated_mismatch = (mitigated_results['label_mismatch'] == True).sum()
    improvement = ((baseline_mismatch - mitigated_mismatch) / max(baseline_mismatch, 1)) * 100

    comparison_data.append({
        'metric': 'Label Mismatches',
        'baseline': baseline_mismatch,
        'mitigated': mitigated_mismatch,
        'improvement_pct': improvement,
        'improvement_direction': 'better' if improvement > 0 else 'worse'
    })

    # Category-specific metrics
    for category in baseline_results['category'].unique():
        baseline_cat = baseline_results[baseline_results['category'] == category]
        mitigated_cat = mitigated_results[mitigated_results['category'] == category]

        baseline_cat_avg = baseline_cat['abs_difference'].mean()
        mitigated_cat_avg = mitigated_cat['abs_difference'].mean()
        improvement = ((baseline_cat_avg - mitigated_cat_avg) / baseline_cat_avg) * 100

        comparison_data.append({
            'metric': f'{category} - Avg Difference',
            'baseline': baseline_cat_avg,
            'mitigated': mitigated_cat_avg,
            'improvement_pct': improvement,
            'improvement_direction': 'better' if improvement > 0 else 'worse'
        })

    comparison_df = pd.DataFrame(comparison_data)

    # Print summary
    print("\nOverall Comparison:")
    print(f"  Avg Difference:    {baseline_avg_diff:.4f} → {mitigated_avg_diff:.4f} ({comparison_df.iloc[0]['improvement_pct']:+.1f}%)")
    print(f"  Max Difference:    {baseline_max_diff:.4f} → {mitigated_max_diff:.4f} ({comparison_df.iloc[1]['improvement_pct']:+.1f}%)")
    print(f"  Biased Pairs:      {baseline_biased} → {mitigated_biased} ({comparison_df.iloc[2]['improvement_pct']:+.1f}%)")
    print(f"  Label Mismatches:  {baseline_mismatch} → {mitigated_mismatch} ({comparison_df.iloc[3]['improvement_pct']:+.1f}%)")

    print("\nCategory-Specific Improvements:")
    for idx in range(4, len(comparison_df)):
        row = comparison_df.iloc[idx]
        print(f"  {row['metric']}: {row['improvement_pct']:+.1f}%")

    print("="*70)

    return comparison_df


def main():
    """
    Run bias mitigation pipeline.
    Usage: python src/mitigation.py
    """
    print("="*70)
    print("FairSense - Bias Mitigation")
    print("="*70)
    print()

    baseline_path = "data/results_baseline.csv"
    print(f"Loading baseline results from: {baseline_path}")

    try:
        baseline_results = pd.read_csv(baseline_path)
        print(f"✓ Loaded {len(baseline_results)} test pairs")
        print()

        # Apply combined mitigation
        mitigated_results = apply_combined_mitigation(baseline_results)

        # Compare before and after
        comparison = compare_before_after(baseline_results, mitigated_results)

        # Save mitigated results
        mitigated_path = "data/results_mitigated.csv"
        mitigated_results.to_csv(mitigated_path, index=False)
        print(f"\n✓ Mitigated results saved to: {mitigated_path}")

        # Save comparison
        comparison_path = "results/metrics/mitigation_comparison.csv"
        comparison.to_csv(comparison_path, index=False)
        print(f"✓ Comparison saved to: {comparison_path}")

        # Check for remaining biased pairs
        remaining_biased = mitigated_results[mitigated_results['abs_difference'] > 0.2]
        if len(remaining_biased) > 0:
            biased_mitigated_path = "data/biased_pairs_mitigated.csv"
            remaining_biased.to_csv(biased_mitigated_path, index=False)
            print(f"✓ Remaining biased pairs saved to: {biased_mitigated_path}")

        print("\n" + "="*70)
        print("BIAS MITIGATION COMPLETE")
        print("="*70)
        print("\nNext steps:")
        print("  1. Review data/results_mitigated.csv")
        print("  2. Calculate fairness metrics on mitigated results")
        print("  3. Generate comparison visualizations")
        print()

    except FileNotFoundError:
        print(f"✗ Error: Could not find {baseline_path}")
        print("Please run bias detection first: python src/bias_detection.py")


if __name__ == "__main__":
    main()
