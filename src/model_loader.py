"""
model_loader.py - Load and initialize pre-trained sentiment analysis models

This module handles loading sentiment analysis models from Hugging Face and
provides functions to make predictions on text input.

Author: FairSense Project
Purpose: AI Bias Detection in Sentiment Analysis
"""

from transformers import pipeline
import torch
from typing import Dict, List, Union


def load_sentiment_model(model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest") -> pipeline:
    """
    Load a pre-trained sentiment analysis model from Hugging Face.

    This function downloads and initializes a sentiment analysis model. On first run,
    it will download the model files (this can take a few minutes). Subsequent runs
    will use the cached version.

    Args:
        model_name (str): The name of the Hugging Face model to load.
                         Default: "cardiffnlp/twitter-roberta-base-sentiment-latest"
                         This is a RoBERTa model trained on Twitter data that returns
                         sentiment labels (negative, neutral, positive) with confidence scores.

    Returns:
        pipeline: A Hugging Face pipeline object ready to make predictions.
                 Use it like: model("text to analyze")

    Raises:
        Exception: If model loading fails (e.g., no internet connection)

    Example:
        >>> model = load_sentiment_model()
        >>> result = model("I love this product!")
        >>> print(result)
        [{'label': 'positive', 'score': 0.95}]
    """
    print(f"Loading sentiment analysis model: {model_name}")
    print("Note: First run will download the model (may take a few minutes)...")

    try:
        # Create a pipeline for sentiment analysis
        # Pipeline is a high-level Hugging Face API that handles:
        # 1. Tokenization (converting text to numbers)
        # 2. Model inference (running the neural network)
        # 3. Post-processing (converting outputs to human-readable format)
        sentiment_pipeline = pipeline(
            "sentiment-analysis",  # Task type
            model=model_name,      # Model identifier
            device=0 if torch.cuda.is_available() else -1  # Use GPU if available, else CPU
        )

        print("✓ Model loaded successfully!")

        # Display device being used
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
    Predict sentiment for one or more text samples.

    Args:
        model (pipeline): The loaded sentiment analysis model from load_sentiment_model()
        text (str or list): A single text string or list of text strings to analyze
        return_all_scores (bool): If True, return scores for all labels (negative, neutral, positive)
                                 If False, return only the top prediction
                                 Default: False

    Returns:
        dict or list: Prediction results containing 'label' and 'score'
                     - If input is a single string, returns a single dict
                     - If input is a list, returns a list of dicts

    Example:
        >>> model = load_sentiment_model()
        >>> result = predict_sentiment(model, "This is amazing!")
        >>> print(result)
        {'label': 'positive', 'score': 0.98}

        >>> # Batch prediction
        >>> texts = ["I love this!", "This is terrible", "It's okay"]
        >>> results = predict_sentiment(model, texts)
        >>> for text, result in zip(texts, results):
        ...     print(f"{text} -> {result['label']} ({result['score']:.2f})")
    """
    try:
        # Run the model on the input text
        predictions = model(text, return_all_scores=return_all_scores)

        # If single text input, unwrap from list
        if isinstance(text, str):
            return predictions[0]

        return predictions

    except Exception as e:
        print(f"✗ Error during prediction: {e}")
        raise


def test_model_with_examples(model: pipeline) -> None:
    """
    Test the sentiment model with a variety of example sentences.

    This function demonstrates how the model works by testing it on clearly
    positive, negative, and neutral sentences. Useful for understanding model
    behavior before running bias tests.

    Args:
        model (pipeline): The loaded sentiment analysis model

    Returns:
        None (prints results to console)

    Example:
        >>> model = load_sentiment_model()
        >>> test_model_with_examples(model)
    """
    print("\n" + "="*70)
    print("TESTING MODEL WITH EXAMPLE SENTENCES")
    print("="*70)

    # Test cases covering different sentiment types
    test_sentences = [
        # Clearly positive
        "This is absolutely wonderful and amazing!",
        "I love this product, it exceeded all my expectations.",

        # Clearly negative
        "This is terrible and I hate it.",
        "Worst experience ever, completely disappointed.",

        # Neutral
        "The meeting is scheduled for tomorrow at 3pm.",
        "The package arrived on Tuesday.",

        # Ambiguous
        "The product is okay, nothing special.",
        "It works but could be better.",
    ]

    print("\nRunning predictions...\n")

    for i, sentence in enumerate(test_sentences, 1):
        # Get prediction with all scores to see model confidence
        result = predict_sentiment(model, sentence, return_all_scores=True)

        # Find the top prediction
        top_prediction = max(result, key=lambda x: x['score'])

        print(f"Test {i}:")
        print(f"  Text: \"{sentence}\"")
        print(f"  Prediction: {top_prediction['label'].upper()}")
        print(f"  Confidence: {top_prediction['score']:.4f}")

        # Show all scores for transparency
        print(f"  All scores: ", end="")
        for score_dict in result:
            print(f"{score_dict['label']}: {score_dict['score']:.4f}  ", end="")
        print("\n")

    print("="*70)
    print("Testing complete!")
    print("="*70)


def main():
    """
    Main function to demonstrate model loading and testing.

    Run this script directly to verify everything is working:
        python src/model_loader.py
    """
    print("="*70)
    print("FairSense - Model Loader Module")
    print("="*70)
    print()

    # Step 1: Load the model
    print("STEP 1: Loading sentiment analysis model...")
    model = load_sentiment_model()
    print()

    # Step 2: Test with example sentences
    print("STEP 2: Testing model with example sentences...")
    test_model_with_examples(model)
    print()

    # Step 3: Interactive test (optional)
    print("="*70)
    print("MODEL READY FOR USE")
    print("="*70)
    print("\nYou can now use this model in other scripts:")
    print("  from src.model_loader import load_sentiment_model, predict_sentiment")
    print("  model = load_sentiment_model()")
    print("  result = predict_sentiment(model, 'Your text here')")
    print()


# This allows the script to be run directly for testing
if __name__ == "__main__":
    main()
