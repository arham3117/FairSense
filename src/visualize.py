"""
visualize.py - Create visualizations for bias analysis

Generates charts to visualize bias patterns and mitigation results.

Author: Muhammad Arham
Course: Introduction to Safety of AI
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Optional


sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_score_distributions(results: pd.DataFrame, group_column: str = 'category',
                             save_path: str = "results/visualizations/score_distributions.png") -> None:
    """
    Create violin plots comparing sentiment score distributions.

    Args:
        results: Bias detection results
        group_column: Column defining groups
        save_path: Where to save the plot
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    data_a = []
    data_b = []
    labels = []

    for category in sorted(results[group_column].unique()):
        cat_data = results[results[group_column] == category]
        data_a.append(cat_data['score_a'].values)
        data_b.append(cat_data['score_b'].values)
        labels.append(category)

    positions_a = np.arange(len(labels)) * 2
    positions_b = positions_a + 0.8

    parts_a = ax.violinplot(data_a, positions=positions_a, widths=0.7,
                            showmeans=True, showmedians=True)
    parts_b = ax.violinplot(data_b, positions=positions_b, widths=0.7,
                            showmeans=True, showmedians=True)

    for pc in parts_a['bodies']:
        pc.set_facecolor('#3498db')
        pc.set_alpha(0.7)
    for pc in parts_b['bodies']:
        pc.set_facecolor('#e74c3c')
        pc.set_alpha(0.7)
    ax.set_xlabel('Test Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sentiment Score', fontsize=12, fontweight='bold')
    ax.set_title('Score Distribution Comparison: Group A vs Group B',
                fontsize=14, fontweight='bold')
    ax.set_xticks((positions_a + positions_b) / 2)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)

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
    Create heatmap showing bias intensity by category.

    Args:
        results: Bias detection results
        save_path: Where to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    category_metrics = results.groupby('category').agg({
        'abs_difference': ['mean', 'max'],
        'label_mismatch': 'sum'
    }).round(4)

    category_metrics.columns = ['Avg Difference', 'Max Difference', 'Label Mismatches']

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
    Create before/after comparison showing mitigation effectiveness.

    Args:
        baseline_results: Original biased predictions
        mitigated_results: After mitigation
        save_path: Where to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    categories = sorted(baseline_results['category'].unique())
    baseline_means = []
    mitigated_means = []
    improvements = []

    for category in categories:
        baseline_cat = baseline_results[baseline_results['category'] == category]
        mitigated_cat = mitigated_results[mitigated_results['category'] == category]

        baseline_mean = baseline_cat['abs_difference'].mean()
        mitigated_mean = mitigated_cat['abs_difference'].mean()

        baseline_means.append(baseline_mean)
        mitigated_means.append(mitigated_mean)

        improvement = ((baseline_mean - mitigated_mean) / baseline_mean) * 100 if baseline_mean > 0 else 0
        improvements.append(improvement)

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax1.bar(x - width/2, baseline_means, width, label='Baseline', color='#e74c3c', alpha=0.8)
    bars2 = ax1.bar(x + width/2, mitigated_means, width, label='Mitigated', color='#2ecc71', alpha=0.8)

    ax1.set_xlabel('Test Category', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Absolute Difference', fontsize=12, fontweight='bold')
    ax1.set_title('Bias Levels: Before vs After Mitigation', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=8)

    colors_improvement = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
    bars3 = ax2.barh(categories, improvements, color=colors_improvement, alpha=0.8)

    ax2.set_xlabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Test Category', fontsize=12, fontweight='bold')
    ax2.set_title('Bias Reduction by Category', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

    for i, (cat, imp) in enumerate(zip(categories, improvements)):
        ax2.text(imp + 2, i, f'{imp:+.1f}%', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Before/after comparison plot saved to: {save_path}")


def plot_bias_by_category(results: pd.DataFrame,
                          save_path: str = "results/visualizations/bias_by_category.png") -> None:
    """
    Create bar chart showing bias levels by category.

    Args:
        results: Bias detection results
        save_path: Where to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    category_bias = results.groupby('category')['abs_difference'].mean().sort_values(ascending=True)

    colors = ['#2ecc71' if x < 0.05 else '#f39c12' if x < 0.15 else '#e74c3c'
              for x in category_bias.values]

    category_bias.plot(kind='barh', color=colors, ax=ax)

    ax.set_xlabel('Average Absolute Difference', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Category', fontsize=12, fontweight='bold')
    ax.set_title('Bias Levels by Test Category', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    for i, v in enumerate(category_bias.values):
        ax.text(v + 0.002, i, f'{v:.4f}', va='center')

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
    Create confusion matrix showing sentiment prediction differences.

    Args:
        results: Bias detection results
        save_path: Where to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    sentiments = ['negative', 'neutral', 'positive']
    confusion_matrix = pd.DataFrame(0, index=sentiments, columns=sentiments)

    for _, row in results.iterrows():
        sent_a = row['sentiment_a']
        sent_b = row['sentiment_b']
        confusion_matrix.loc[sent_a, sent_b] += 1

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


def generate_all_visualizations(baseline_results: pd.DataFrame,
                                mitigated_results: Optional[pd.DataFrame] = None,
                                metrics: Optional[dict] = None) -> None:
    """
    Generate all visualizations for the project.

    Args:
        baseline_results: Original bias test results
        mitigated_results: Post-mitigation results (optional)
        metrics: Fairness metrics (optional)
    """
    print("\nGenerating visualizations...")
    print("-" * 70)

    plot_score_distributions(baseline_results)
    plot_bias_heatmap(baseline_results)
    plot_bias_by_category(baseline_results)
    plot_sentiment_confusion_matrix(baseline_results)

    print("-" * 70)
    print("✓ All visualizations generated successfully!")


def main():
    """
    Generate visualizations from baseline results.
    Usage: python src/visualize.py
    """
    print("="*70)
    print("FairSense - Visualization Generator")
    print("="*70)
    print()

    results_path = "data/results_baseline.csv"
    print(f"Loading baseline results from: {results_path}")

    try:
        results = pd.read_csv(results_path)
        print(f"✓ Loaded {len(results)} test pairs")

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
