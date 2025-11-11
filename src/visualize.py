"""
visualize.py - Create visualizations for bias analysis

This module generates charts and plots to visualize bias patterns:
- Score distribution comparisons
- Bias heatmaps
- Before/after mitigation comparisons
- Category-specific bias breakdowns

Author: FairSense Project
Purpose: AI Bias Detection in Sentiment Analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Optional


# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_score_distributions(results: pd.DataFrame, group_column: str = 'category',
                             save_path: str = "results/visualizations/score_distributions.png") -> None:
    """
    Create distribution plots comparing sentiment scores across groups.

    Shows how sentiment scores are distributed for different demographic groups,
    helping identify systematic biases.

    Args:
        results (pd.DataFrame): Bias detection results
        group_column (str): Column defining groups to compare
        save_path (str): Where to save the plot

    Returns:
        None (saves plot to file and displays)
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Prepare data for violin plot
    data_a = []
    data_b = []
    labels = []

    for category in sorted(results[group_column].unique()):
        cat_data = results[results[group_column] == category]
        data_a.append(cat_data['score_a'].values)
        data_b.append(cat_data['score_b'].values)
        labels.append(category)

    # Create positions for violin plots
    positions_a = np.arange(len(labels)) * 2
    positions_b = positions_a + 0.8

    # Create violin plots
    parts_a = ax.violinplot(data_a, positions=positions_a, widths=0.7,
                            showmeans=True, showmedians=True)
    parts_b = ax.violinplot(data_b, positions=positions_b, widths=0.7,
                            showmeans=True, showmedians=True)

    # Color the violin plots
    for pc in parts_a['bodies']:
        pc.set_facecolor('#3498db')
        pc.set_alpha(0.7)
    for pc in parts_b['bodies']:
        pc.set_facecolor('#e74c3c')
        pc.set_alpha(0.7)

    # Customize plot
    ax.set_xlabel('Test Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sentiment Score', fontsize=12, fontweight='bold')
    ax.set_title('Score Distribution Comparison: Group A vs Group B',
                fontsize=14, fontweight='bold')
    ax.set_xticks((positions_a + positions_b) / 2)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#3498db', alpha=0.7, label='Group A'),
                      Patch(facecolor='#e74c3c', alpha=0.7, label='Group B')]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Score distribution plot saved to: {save_path}")


def plot_bias_heatmap(results: pd.DataFrame, save_path: str = "results/visualizations/bias_heatmap.png") -> None:
    """
    Create heatmap showing bias intensity across different test categories.

    Helps visualize which types of bias are most severe and where
    mitigation efforts should focus.

    Args:
        results (pd.DataFrame): Bias detection results
        save_path (str): Where to save the plot

    Returns:
        None (saves plot to file and displays)
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Calculate average bias metrics per category
    category_metrics = results.groupby('category').agg({
        'abs_difference': ['mean', 'max'],
        'label_mismatch': 'sum'
    }).round(4)

    category_metrics.columns = ['Avg Difference', 'Max Difference', 'Label Mismatches']

    # Create heatmap
    sns.heatmap(category_metrics, annot=True, fmt='.4f', cmap='YlOrRd',
                cbar_kws={'label': 'Bias Intensity'}, ax=ax, linewidths=0.5)

    ax.set_title('Bias Intensity Heatmap by Category', fontsize=14, fontweight='bold')
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Category', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Bias heatmap saved to: {save_path}")


def plot_before_after_comparison(baseline_results: pd.DataFrame,
                                 mitigated_results: pd.DataFrame,
                                 save_path: str = "results/visualizations/before_after_comparison.png") -> None:
    """
    Create side-by-side comparison of bias before and after mitigation.

    Shows the effectiveness of mitigation strategies with:
    - Baseline bias scores
    - Mitigated bias scores
    - Improvement bars

    Args:
        baseline_results (pd.DataFrame): Original biased predictions
        mitigated_results (pd.DataFrame): After mitigation
        save_path (str): Where to save the plot

    Returns:
        None (saves plot to file and displays)
    """
    # TODO: Calculate metrics for both datasets
    # TODO: Create grouped bar chart
    # TODO: Add improvement percentage labels
    # TODO: Save and display plot

    pass


def plot_bias_by_category(results: pd.DataFrame,
                          save_path: str = "results/visualizations/bias_by_category.png") -> None:
    """
    Create bar chart showing bias levels across different test categories.

    Compares bias in:
    - Gender tests
    - Occupational tests
    - Name-based tests

    Args:
        results (pd.DataFrame): Bias detection results
        save_path (str): Where to save the plot

    Returns:
        None (saves plot to file and displays)
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate average absolute difference by category
    category_bias = results.groupby('category')['abs_difference'].mean().sort_values(ascending=True)

    # Color code by severity
    colors = ['#2ecc71' if x < 0.05 else '#f39c12' if x < 0.15 else '#e74c3c'
              for x in category_bias.values]

    # Create horizontal bar chart
    category_bias.plot(kind='barh', color=colors, ax=ax)

    ax.set_xlabel('Average Absolute Difference', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Category', fontsize=12, fontweight='bold')
    ax.set_title('Bias Levels by Test Category', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for i, v in enumerate(category_bias.values):
        ax.text(v + 0.002, i, f'{v:.4f}', va='center')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Low bias (<0.05)'),
        Patch(facecolor='#f39c12', label='Moderate bias (0.05-0.15)'),
        Patch(facecolor='#e74c3c', label='High bias (>0.15)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Bias by category plot saved to: {save_path}")


def plot_sentiment_confusion_matrix(results: pd.DataFrame,
                                    save_path: str = "results/visualizations/confusion_matrix.png") -> None:
    """
    Create confusion matrix showing how sentiment predictions differ between groups.

    For paired sentences (e.g., he/she versions), shows how often:
    - Both get same sentiment
    - Group A gets positive, Group B gets neutral
    - etc.

    Args:
        results (pd.DataFrame): Bias detection results with paired predictions
        save_path (str): Where to save the plot

    Returns:
        None (saves plot to file and displays)
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create confusion matrix
    sentiments = ['negative', 'neutral', 'positive']
    confusion_matrix = pd.DataFrame(0, index=sentiments, columns=sentiments)

    for _, row in results.iterrows():
        sent_a = row['sentiment_a']
        sent_b = row['sentiment_b']
        confusion_matrix.loc[sent_a, sent_b] += 1

    # Create heatmap
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                cbar_kws={'label': 'Count'}, ax=ax, linewidths=1)

    ax.set_title('Sentiment Prediction Confusion Matrix\n(Group A vs Group B)',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Group B Sentiment', fontsize=12, fontweight='bold')
    ax.set_ylabel('Group A Sentiment', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved to: {save_path}")


def plot_fairness_metrics_dashboard(metrics: dict,
                                    save_path: str = "results/visualizations/fairness_dashboard.png") -> None:
    """
    Create comprehensive dashboard showing all fairness metrics.

    Multi-panel visualization including:
    - Demographic parity scores
    - Equalized odds
    - Bias severity gauge
    - Category breakdowns

    Args:
        metrics (dict): Fairness metrics from fairness_metrics.py
        save_path (str): Where to save the plot

    Returns:
        None (saves plot to file and displays)
    """
    # TODO: Create multi-panel figure
    # TODO: Add gauge chart for overall bias severity
    # TODO: Add bar charts for individual metrics
    # TODO: Include summary statistics
    # TODO: Save and display plot

    pass


def generate_all_visualizations(baseline_results: pd.DataFrame,
                                mitigated_results: Optional[pd.DataFrame] = None,
                                metrics: Optional[dict] = None) -> None:
    """
    Generate complete set of visualizations for the project.

    Creates all plots needed for the final report:
    - Distribution plots
    - Heatmaps
    - Category comparisons
    - Before/after comparisons (if mitigation applied)
    - Fairness dashboard (if metrics provided)

    Args:
        baseline_results (pd.DataFrame): Original bias test results
        mitigated_results (pd.DataFrame, optional): Post-mitigation results
        metrics (dict, optional): Fairness metrics

    Returns:
        None (saves all plots to results/visualizations/)
    """
    print("\nGenerating visualizations...")
    print("-" * 70)

    # Generate core visualizations
    plot_score_distributions(baseline_results)
    plot_bias_heatmap(baseline_results)
    plot_bias_by_category(baseline_results)
    plot_sentiment_confusion_matrix(baseline_results)

    print("-" * 70)
    print("✓ All visualizations generated successfully!")


def main():
    """
    Main function to demonstrate visualization creation.

    Run this script directly:
        python src/visualize.py
    """
    print("="*70)
    print("FairSense - Visualization Generator")
    print("="*70)
    print("\nThis module creates comprehensive visualizations including:")
    print("  - Score distribution comparisons")
    print("  - Bias intensity heatmap")
    print("  - Bias levels by category")
    print("  - Sentiment confusion matrix")
    print()

    # Load baseline results
    results_path = "data/results_baseline.csv"
    print(f"Loading baseline results from: {results_path}")

    try:
        results = pd.read_csv(results_path)
        print(f"✓ Loaded {len(results)} test pairs")

        # Generate all visualizations
        generate_all_visualizations(results)

        print("\n" + "="*70)
        print("VISUALIZATION GENERATION COMPLETE")
        print("="*70)
        print("\nGenerated visualizations in results/visualizations/:")
        print("  - score_distributions.png")
        print("  - bias_heatmap.png")
        print("  - bias_by_category.png")
        print("  - confusion_matrix.png")
        print("\nThese visualizations can be included in your final report.")
        print()

    except FileNotFoundError:
        print(f"✗ Error: Could not find {results_path}")
        print("Please run bias detection first: python src/bias_detection.py")


if __name__ == "__main__":
    main()
