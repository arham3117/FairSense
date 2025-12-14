"""
model_loader.py - Load and run sentiment analysis models

Loads pre-trained models from Hugging Face and makes sentiment predictions.

Author: Muhammad Arham
Course: Introduction to Safety of AI
"""

from transformers import pipeline
import torch
from typing import Dict, List, Union


def load_sentiment_model(model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest") -> pipeline:
    """
    Load a sentiment analysis model from Hugging Face.

    Args:
        model_name: Name of the model (default: RoBERTa sentiment model)

    Returns:
        pipeline: Model ready to make predictions

    Example:
        model = load_sentiment_model()
        result = model("I love this!")
        # Returns: [{'label': 'positive', 'score': 0.95}]
    """
    print(f"Loading sentiment analysis model: {model_name}")
    print("Note: First run will download the model (may take a few minutes)...")

    try:
        # Create sentiment analysis pipeline
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1  # GPU if available, else CPU
        )

        print("✓ Model loaded successfully!")
        device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        print(f"Using device: {device}")

        return sentiment_pipeline

    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure you have internet connection (for first-time download)")
        print("2. Check that transformers and torch are installed")
        print("3. Try running: pip install --upgrade transformers torch")
        raise


def predict_sentiment(model: pipeline, text: Union[str, List[str]],
                      return_all_scores: bool = False) -> Union[Dict, List[Dict]]:
    """
    Predict sentiment for text.

    Args:
        model: Loaded sentiment model
        text: Single string or list of strings
        return_all_scores: If True, return scores for all labels

    Returns:
        Prediction with label and score

    Example:
        result = predict_sentiment(model, "This is amazing!")
        # Returns: {'label': 'positive', 'score': 0.98}
    """
    try:
        predictions = model(text, return_all_scores=return_all_scores)

        # Unwrap single predictions
        if isinstance(text, str):
            return predictions[0]

        return predictions

    except Exception as e:
        print(f"✗ Error during prediction: {e}")
        raise


def test_model_with_examples(model: pipeline) -> None:
    """
    Test model with example sentences to verify it works correctly.

    Args:
        model: Loaded sentiment model

    Returns:
        None (prints results)
    """
    print("\n" + "="*70)
    print("TESTING MODEL WITH EXAMPLE SENTENCES")
    print("="*70)

    test_sentences = [
        "This is absolutely wonderful and amazing!",
        "I love this product, it exceeded all my expectations.",
        "This is terrible and I hate it.",
        "Worst experience ever, completely disappointed.",
        "The meeting is scheduled for tomorrow at 3pm.",
        "The package arrived on Tuesday.",
        "The product is okay, nothing special.",
        "It works but could be better.",
    ]

    print("\nRunning predictions...\n")

    for i, sentence in enumerate(test_sentences, 1):
        result = predict_sentiment(model, sentence, return_all_scores=True)
        top_prediction = max(result, key=lambda x: x['score'])

        print(f"Test {i}:")
        print(f"  Text: \"{sentence}\"")
        print(f"  Prediction: {top_prediction['label'].upper()}")
        print(f"  Confidence: {top_prediction['score']:.4f}")
        print(f"  All scores: ", end="")
        for score_dict in result:
            print(f"{score_dict['label']}: {score_dict['score']:.4f}  ", end="")
        print("\n")

    print("="*70)
    print("Testing complete!")
    print("="*70)


def main():
    """
    Run model loading and testing.
    Usage: python src/model_loader.py
    """
    print("="*70)
    print("FairSense - Model Loader Module")
    print("="*70)
    print()

    print("STEP 1: Loading sentiment analysis model...")
    model = load_sentiment_model()
    print()

    print("STEP 2: Testing model with example sentences...")
    test_model_with_examples(model)
    print()

    print("="*70)
    print("MODEL READY FOR USE")
    print("="*70)
    print("\nYou can now use this model in other scripts:")
    print("  from src.model_loader import load_sentiment_model, predict_sentiment")
    print("  model = load_sentiment_model()")
    print("  result = predict_sentiment(model, 'Your text here')")
    print()


if __name__ == "__main__":
    main()
