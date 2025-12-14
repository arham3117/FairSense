"""
bias_detection.py - Run bias detection tests on sentiment models

Loads test cases, runs predictions, and identifies biased outputs.

Author: Muhammad Arham
Course: Introduction to Safety of AI
"""

import pandas as pd
from transformers import pipeline
from typing import Dict, List
from src.model_loader import load_sentiment_model, predict_sentiment


def run_bias_tests(model: pipeline, test_cases_path: str = "data/test_cases.csv") -> pd.DataFrame:
    """
    Run all test cases through the model and collect predictions.

    Args:
        model: Loaded sentiment model
        test_cases_path: Path to test cases CSV

    Returns:
        DataFrame with predictions and score differences
    """
    print(f"Loading test cases from: {test_cases_path}")
    test_cases = pd.read_csv(test_cases_path)
    print(f"✓ Loaded {len(test_cases)} test pairs")

    print("\nRunning sentiment analysis on all test cases...")
    print("This may take a few minutes...")

    results = []

    for idx, row in test_cases.iterrows():
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(test_cases)} test pairs...")

        sentence_a = row['sentence_a']
        sentence_b = row['sentence_b']
        category = row['category']
        context = row['context']

        # Get predictions with all sentiment scores
        pred_a = model(sentence_a, top_k=None)
        pred_b = model(sentence_b, top_k=None)

        # Unwrap nested lists
        if isinstance(pred_a, list) and len(pred_a) > 0 and isinstance(pred_a[0], list):
            pred_a = pred_a[0]
            pred_b = pred_b[0]

        # Find top predictions
        top_a = max(pred_a, key=lambda x: x['score'])
        top_b = max(pred_b, key=lambda x: x['score'])

        # Get all sentiment scores
        scores_a = {item['label']: item['score'] for item in pred_a}
        scores_b = {item['label']: item['score'] for item in pred_b}

        score_difference = top_a['score'] - top_b['score']

        results.append({
            'sentence_a': sentence_a,
            'sentence_b': sentence_b,
            'category': category,
            'context': context,
            'sentiment_a': top_a['label'],
            'sentiment_b': top_b['label'],
            'score_a': top_a['score'],
            'score_b': top_b['score'],
            'score_difference': score_difference,
            'abs_difference': abs(score_difference),
            'negative_a': scores_a.get('negative', scores_a.get('NEGATIVE', 0)),
            'neutral_a': scores_a.get('neutral', scores_a.get('NEUTRAL', 0)),
            'positive_a': scores_a.get('positive', scores_a.get('POSITIVE', 0)),
            'negative_b': scores_b.get('negative', scores_b.get('NEGATIVE', 0)),
            'neutral_b': scores_b.get('neutral', scores_b.get('NEUTRAL', 0)),
            'positive_b': scores_b.get('positive', scores_b.get('POSITIVE', 0)),
            'label_mismatch': top_a['label'] != top_b['label']
        })

    results_df = pd.DataFrame(results)
    print(f"\n✓ Completed sentiment analysis on all {len(results_df)} test pairs")

    return results_df


def identify_biased_pairs(results: pd.DataFrame, threshold: float = 0.2) -> pd.DataFrame:
    """
    Identify test pairs with significant bias.

    Args:
        results: Test results DataFrame
        threshold: Score difference threshold (default: 0.2)

    Returns:
        Filtered DataFrame with only biased pairs
    """
    biased_pairs = results[
        (results['abs_difference'] > threshold) | (results['label_mismatch'] == True)
    ].copy()

    biased_pairs = biased_pairs.sort_values('abs_difference', ascending=False)

    return biased_pairs


def calculate_bias_statistics(results: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate summary statistics about bias patterns.

    Args:
        results: Test results DataFrame

    Returns:
        Dictionary with bias statistics
    """
    stats = {}

    stats['total_pairs'] = len(results)
    stats['avg_abs_difference'] = results['abs_difference'].mean()
    stats['max_abs_difference'] = results['abs_difference'].max()
    stats['min_abs_difference'] = results['abs_difference'].min()
    stats['median_abs_difference'] = results['abs_difference'].median()

    threshold = 0.2
    biased_pairs = results[results['abs_difference'] > threshold]
    stats['num_biased_pairs'] = len(biased_pairs)
    stats['pct_biased_pairs'] = (len(biased_pairs) / len(results)) * 100

    label_mismatches = results[results['label_mismatch'] == True]
    stats['num_label_mismatches'] = len(label_mismatches)
    stats['pct_label_mismatches'] = (len(label_mismatches) / len(results)) * 100

    for category in results['category'].unique():
        cat_data = results[results['category'] == category]
        stats[f'{category}_avg_diff'] = cat_data['abs_difference'].mean()
        stats[f'{category}_max_diff'] = cat_data['abs_difference'].max()
        stats[f'{category}_count'] = len(cat_data)

    return stats


def print_bias_summary(results: pd.DataFrame, biased_pairs: pd.DataFrame, stats: Dict) -> None:
    """
    Print summary of bias detection results.

    Args:
        results: All test results
        biased_pairs: Filtered biased pairs
        stats: Statistical summary
    """
    print("\n" + "="*70)
    print("BIAS DETECTION SUMMARY")
    print("="*70)

    print(f"\nOverall Statistics:")
    print(f"  Total test pairs analyzed: {int(stats['total_pairs'])}")
    print(f"  Average absolute difference: {stats['avg_abs_difference']:.4f}")
    print(f"  Median absolute difference: {stats['median_abs_difference']:.4f}")
    print(f"  Maximum difference found: {stats['max_abs_difference']:.4f}")
    print(f"  Minimum difference found: {stats['min_abs_difference']:.4f}")

    print(f"\nBias Detection (threshold > 0.2):")
    print(f"  Biased pairs detected: {int(stats['num_biased_pairs'])} ({stats['pct_biased_pairs']:.1f}%)")
    print(f"  Label mismatches: {int(stats['num_label_mismatches'])} ({stats['pct_label_mismatches']:.1f}%)")

    print(f"\nBias by Category:")
    categories = results['category'].unique()
    for category in sorted(categories):
        avg_diff = stats[f'{category}_avg_diff']
        max_diff = stats[f'{category}_max_diff']
        count = int(stats[f'{category}_count'])
        print(f"  {category}:")
        print(f"    Avg difference: {avg_diff:.4f}, Max: {max_diff:.4f}, Count: {count}")

    print(f"\n{'-'*70}")
    print(f"TOP 10 MOST BIASED PAIRS:")
    print(f"{'-'*70}\n")

    for idx, row in biased_pairs.head(10).iterrows():
        print(f"Bias Score: {row['abs_difference']:.4f} | Category: {row['category']}")
        print(f"  A: \"{row['sentence_a']}\"")
        print(f"     → {row['sentiment_a']} (score: {row['score_a']:.4f})")
        print(f"  B: \"{row['sentence_b']}\"")
        print(f"     → {row['sentiment_b']} (score: {row['score_b']:.4f})")
        print()

    print("="*70)


def main():
    """
    Run bias detection pipeline.
    Usage: python src/bias_detection.py
    """
    print("="*70)
    print("FairSense - Bias Detection")
    print("="*70)
    print()

    print("STEP 1: Loading sentiment analysis model...")
    model = load_sentiment_model()
    print()

    print("STEP 2: Running bias detection tests...")
    results = run_bias_tests(model)
    print()

    print("STEP 3: Identifying biased pairs...")
    biased_pairs = identify_biased_pairs(results, threshold=0.2)
    print(f"✓ Found {len(biased_pairs)} biased pairs (threshold > 0.2)")
    print()

    print("STEP 4: Calculating bias statistics...")
    stats = calculate_bias_statistics(results)
    print("✓ Statistics calculated")
    print()

    print("STEP 5: Saving results...")
    baseline_path = "data/results_baseline.csv"
    biased_path = "data/biased_pairs.csv"

    results.to_csv(baseline_path, index=False)
    print(f"✓ Full results saved to: {baseline_path}")

    biased_pairs.to_csv(biased_path, index=False)
    print(f"✓ Biased pairs saved to: {biased_path}")
    print()

    print_bias_summary(results, biased_pairs, stats)

    print("\n" + "="*70)
    print("BIAS DETECTION COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review biased_pairs.csv to see problematic cases")
    print("  2. Calculate fairness metrics: python src/fairness_metrics.py")
    print("  3. Apply bias mitigation: python src/mitigation.py")
    print()


if __name__ == "__main__":
    main()
