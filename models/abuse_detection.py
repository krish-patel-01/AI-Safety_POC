"""
Abuse Language Detection Model
Detects harmful, threatening, or inappropriate content using pretrained toxic-bert model
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')


# Global model cache
_abuse_model = None
_abuse_tokenizer = None


def load_abuse_detection_model():
    """
    Load pretrained toxicity detection model
    Uses unitary/toxic-bert for high accuracy toxic content detection
    
    Returns:
        Tuple of (model, tokenizer)
    """
    global _abuse_model, _abuse_tokenizer
    
    if _abuse_model is None:
        print("Loading abuse detection model (unitary/toxic-bert)...")
        model_name = "unitary/toxic-bert"
        _abuse_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _abuse_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _abuse_model.eval()
        print("✓ Abuse detection model loaded successfully")
    
    return _abuse_model, _abuse_tokenizer


def detect_abuse(text: str, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Detect abusive language in text
    
    Args:
        text: Input text to analyze
        threshold: Classification threshold (default: 0.5)
        
    Returns:
        Dictionary containing:
            - is_abusive: Boolean indicating if text is abusive
            - toxicity_scores: Dict of individual toxicity category scores
            - max_toxicity: Highest toxicity score
            - flagged_categories: List of categories exceeding threshold
    """
    model, tokenizer = load_abuse_detection_model()
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.sigmoid(outputs.logits)[0]
    
    # toxic-bert predicts 6 categories: toxic, severe_toxic, obscene, threat, insult, identity_hate
    categories = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    
    toxicity_scores = {
        category: float(predictions[i])
        for i, category in enumerate(categories)
    }
    
    max_toxicity = max(toxicity_scores.values())
    flagged_categories = [cat for cat, score in toxicity_scores.items() if score > threshold]
    is_abusive = len(flagged_categories) > 0
    
    return {
        "is_abusive": is_abusive,
        "toxicity_scores": toxicity_scores,
        "max_toxicity": max_toxicity,
        "flagged_categories": flagged_categories,
        "confidence": max_toxicity if is_abusive else (1.0 - max_toxicity)
    }

if __name__ == "__main__":
    # Test the abuse detection
    test_messages = [
        "Hello, how are you?",
        "You're an idiot and nobody likes you!",
        "I hate you so much, you're worthless!",
        "This is a nice day for a walk in the park.",
        "Go kill yourself, loser!"
    ]
    
    print("Testing Abuse Detection Model\n" + "="*50)
    for msg in test_messages:
        result = detect_abuse(msg)
        print(f"\nMessage: {msg}")
        print(f"Is Abusive: {result['is_abusive']}")
        print(f"Max Toxicity: {result['max_toxicity']:.4f}")
        print(f"Flagged Categories: {result['flagged_categories']}")
