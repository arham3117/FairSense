"""
bias_detection.py - Run bias detection tests on sentiment models

This module loads test cases, runs them through the sentiment model,
and collects results for analysis.

Author: FairSense Project
Purpose: AI Bias Detection in Sentiment Analysis
"""

import pandas as pd
from transformers import pipeline
from typing import Dict, List
from src.model_loader import load_sentiment_model, predict_sentiment


def run_bias_tests(model: pipeline, test_cases_path: str = "data/test_cases.csv") -> pd.DataFrame:
    """
    Run all test cases through the sentiment model and collect results.

    Args:
        model (pipeline): Loaded sentiment analysis model
        test_cases_path (str): Path to CSV file containing test cases

    Returns:
        pd.DataFrame: Results with sentiment predictions for each test case

    Example results structure:
        | sentence_a | sentence_b | sentiment_a | score_a | sentiment_b | score_b | difference |
        |------------|------------|-------------|---------|-------------|---------|------------|
        | He is...   | She is...  | positive    | 0.89    | neutral     | 0.65    | 0.24       |
    """
    print(f"Loading test cases from: {test_cases_path}")
    test_cases = pd.read_csv(test_cases_path)
    print(f"✓ Loaded {len(test_cases)} test pairs")

    print("\nRunning sentiment analysis on all test cases...")
    print("This may take a few minutes...")

    results = []

    # Process each test pair
    for idx, row in test_cases.iterrows():
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(test_cases)} test pairs...")

        sentence_a = row['sentence_a']
        sentence_b = row['sentence_b']
        category = row['category']
        context = row['context']

        # Get predictions with all scores (top_k=None returns all labels)
        # Note: Using top_k instead of deprecated return_all_scores
        pred_a = model(sentence_a, top_k=None)
        pred_b = model(sentence_b, top_k=None)

        # The model returns a list of lists, get the first element
        if isinstance(pred_a, list) and len(pred_a) > 0 and isinstance(pred_a[0], list):
            pred_a = pred_a[0]
            pred_b = pred_b[0]

        # Extract sentiment labels and scores
        # Find the top prediction for each
        top_a = max(pred_a, key=lambda x: x['score'])
        top_b = max(pred_b, key=lambda x: x['score'])

        # Get scores for each sentiment type
        scores_a = {item['label']: item['score'] for item in pred_a}
        scores_b = {item['label']: item['score'] for item in pred_b}

        # Calculate the difference in top scores
        score_difference = top_a['score'] - top_b['score']

        # Store detailed results
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
            # Store individual sentiment scores for detailed analysis
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
    Identify test case pairs that show significant bias.

    A pair is considered biased if the sentiment score difference exceeds
    the threshold (default: 0.2 or 20%).

    Args:
        results (pd.DataFrame): Results from run_bias_tests()
        threshold (float): Minimum score difference to flag as biased

    Returns:
        pd.DataFrame: Filtered results showing only biased pairs
    """
    # Filter pairs where absolute difference exceeds threshold OR labels don't match
    biased_pairs = results[
        (results['abs_difference'] > threshold) | (results['label_mismatch'] == True)
    ].copy()

    # Sort by absolute difference (largest bias first)
    biased_pairs = biased_pairs.sort_values('abs_difference', ascending=False)

    return biased_pairs


def calculate_bias_statistics(results: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate summary statistics about bias patterns.

    Returns metrics like:
    - Average score difference
    - Percentage of biased pairs
    - Bias by category (gender, occupation, name)

    Args:
        results (pd.DataFrame): Results from run_bias_tests()

    Returns:
        Dict[str, float]: Statistical summary of bias patterns
    """
    stats = {}

    # Overall statistics
    stats['total_pairs'] = len(results)
    stats['avg_abs_difference'] = results['abs_difference'].mean()
    stats['max_abs_difference'] = results['abs_difference'].max()
    stats['min_abs_difference'] = results['abs_difference'].min()
    stats['median_abs_difference'] = results['abs_difference'].median()

    # Bias identification
    threshold = 0.2
    biased_pairs = results[results['abs_difference'] > threshold]
    stats['num_biased_pairs'] = len(biased_pairs)
    stats['pct_biased_pairs'] = (len(biased_pairs) / len(results)) * 100

    # Label mismatch statistics
    label_mismatches = results[results['label_mismatch'] == True]
    stats['num_label_mismatches'] = len(label_mismatches)
    stats['pct_label_mismatches'] = (len(label_mismatches) / len(results)) * 100

    # Per-category statistics
    category_stats = results.groupby('category').agg({
        'abs_difference': ['mean', 'max', 'count'],
        'label_mismatch': 'sum'
    })

    for category in results['category'].unique():
        cat_data = results[results['category'] == category]
        stats[f'{category}_avg_diff'] = cat_data['abs_difference'].mean()
        stats[f'{category}_max_diff'] = cat_data['abs_difference'].max()
        stats[f'{category}_count'] = len(cat_data)

    return stats


def print_bias_summary(results: pd.DataFrame, biased_pairs: pd.DataFrame, stats: Dict) -> None:
    """
    Print comprehensive summary of bias detection results.

    Args:
        results (pd.DataFrame): All test results
        biased_pairs (pd.DataFrame): Filtered biased pairs
        stats (Dict): Statistical summary

    Returns:
        None
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
    Main function to run bias detection pipeline.

    Run this script directly:
        python src/bias_detection.py
    """
    print("="*70)
    print("FairSense - Bias Detection")
    print("="*70)
    print("\nThis module will:")
    print("  1. Load the sentiment analysis model")
    print("  2. Run all test cases through the model")
    print("  3. Identify biased predictions")
    print("  4. Save results for analysis")
    print()

    # Step 1: Load model
    print("STEP 1: Loading sentiment analysis model...")
    model = load_sentiment_model()
    print()

    # Step 2: Run bias tests
    print("STEP 2: Running bias detection tests...")
    results = run_bias_tests(model)
    print()

    # Step 3: Identify biased pairs
    print("STEP 3: Identifying biased pairs...")
    biased_pairs = identify_biased_pairs(results, threshold=0.2)
    print(f"✓ Found {len(biased_pairs)} biased pairs (threshold > 0.2)")
    print()

    # Step 4: Calculate statistics
    print("STEP 4: Calculating bias statistics...")
    stats = calculate_bias_statistics(results)
    print("✓ Statistics calculated")
    print()

    # Step 5: Save results
    print("STEP 5: Saving results...")
    baseline_path = "data/results_baseline.csv"
    biased_path = "data/biased_pairs.csv"

    results.to_csv(baseline_path, index=False)
    print(f"✓ Full results saved to: {baseline_path}")

    biased_pairs.to_csv(biased_path, index=False)
    print(f"✓ Biased pairs saved to: {biased_path}")
    print()

    # Step 6: Print summary
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
