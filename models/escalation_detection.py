"""
Escalation Pattern Recognition Model
Detects when conversations are becoming emotionally dangerous through pattern analysis
"""
from transformers import pipeline
from typing import Dict, Any, List, Tuple
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


# Global model cache
_sentiment_model = None
_sentiment_pipeline = None


def load_escalation_detection_model():
    """
    Load pretrained sentiment analysis model for escalation detection
    Uses cardiffnlp/twitter-roberta-base-sentiment-latest for accurate sentiment tracking
    
    Returns:
        Tuple of (model, pipeline)
    """
    global _sentiment_model, _sentiment_pipeline
    
    if _sentiment_pipeline is None:
        print("Loading escalation detection model (sentiment analysis)...")
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            return_all_scores=True
        )
        
        print("✓ Escalation detection model loaded successfully")
    
    return _sentiment_pipeline


def analyze_message_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyze sentiment of a single message
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary with sentiment scores and labels
    """
    sentiment_pipeline = load_escalation_detection_model()
    
    # Get sentiment predictions
    sentiment_results = sentiment_pipeline(text)[0]
    
    # Convert to dictionary
    sentiment_scores = {
        item['label'].lower(): item['score']
        for item in sentiment_results
    }
    
    # Determine dominant sentiment
    dominant = max(sentiment_results, key=lambda x: x['score'])
    
    # Calculate negativity score (for escalation detection)
    negativity_score = sentiment_scores.get('negative', 0.0)
    
    return {
        "sentiment_scores": sentiment_scores,
        "dominant_sentiment": dominant['label'].lower(),
        "confidence": dominant['score'],
        "negativity_score": negativity_score
    }


def calculate_sentiment_change(sentiments: List[float]) -> Tuple[float, str]:
    """
    Calculate sentiment change rate and trend
    
    Args:
        sentiments: List of sentiment scores (chronological)
        
    Returns:
        Tuple of (change_rate, trend_description)
    """
    if len(sentiments) < 2:
        return 0.0, "INSUFFICIENT_DATA"
    
    # Calculate linear trend
    n = len(sentiments)
    x = list(range(n))
    
    # linear regression
    x_mean = sum(x) / n
    y_mean = sum(sentiments) / n
    
    numerator = sum((x[i] - x_mean) * (sentiments[i] - y_mean) for i in range(n))
    denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
    
    if denominator == 0:
        slope = 0.0
    else:
        slope = numerator / denominator
    
    # Interpret trend
    if slope > 0.1:
        trend = "ESCALATING"
    elif slope < -0.1:
        trend = "DE-ESCALATING"
    else:
        trend = "STABLE"
    
    return slope, trend


def detect_repetitive_negative_patterns(messages: List[str]) -> Dict[str, Any]:
    """
    Detect repetitive negative patterns in messages
    
    Args:
        messages: List of message texts
        
    Returns:
        Dictionary with pattern analysis
    """
    if not messages:
        return {
            "has_repetitive_pattern": False,
            "repetition_count": 0,
            "common_words": []
        }
    
    # Tokenize and count words
    negative_words = [
        "hate", "angry", "mad", "upset", "frustrated", "annoyed",
        "irritated", "furious", "rage", "hostile", "aggressive"
    ]
    
    word_counts = Counter()
    for msg in messages:
        words = msg.lower().split()
        for word in words:
            if word in negative_words:
                word_counts[word] += 1
    
    # Check for repetition
    most_common = word_counts.most_common(3)
    has_repetitive_pattern = any(count >= 3 for _, count in most_common)
    
    return {
        "has_repetitive_pattern": has_repetitive_pattern,
        "repetition_count": sum(word_counts.values()),
        "common_negative_words": [word for word, count in most_common if count >= 2]
    }


def detect_escalation(messages: List[str], window_size: int = 5) -> Dict[str, Any]:
    """
    Detect conversation escalation patterns
    
    Args:
        messages: List of message texts in chronological order
        window_size: Number of recent messages to analyze
        
    Returns:
        Dictionary containing:
            - is_escalating: Boolean indicating escalation detected
            - escalation_score: Score from 0.0 to 1.0
            - severity: Escalation severity level
            - trend: Trend description
            - indicators: List of escalation indicators
            - sentiment_trajectory: List of sentiment scores over time
    """
    if not messages:
        return {
            "is_escalating": False,
            "escalation_score": 0.0,
            "severity": "NONE",
            "trend": "NO_DATA",
            "indicators": [],
            "sentiment_trajectory": []
        }
    
    # Analyze recent messages
    recent_messages = messages[-window_size:] if len(messages) > window_size else messages
    
    # Get sentiment for each message
    sentiment_results = [analyze_message_sentiment(msg) for msg in recent_messages]
    negativity_scores = [s['negativity_score'] for s in sentiment_results]
    
    # Calculate trend
    change_rate, trend = calculate_sentiment_change(negativity_scores)
    
    # Check for repetitive patterns
    pattern_analysis = detect_repetitive_negative_patterns(recent_messages)
    
    # Calculate escalation score
    avg_negativity = sum(negativity_scores) / len(negativity_scores)
    recent_negativity = negativity_scores[-1] if negativity_scores else 0.0
    
    # Factors contributing to escalation score
    trend_factor = max(0.0, change_rate) * 2.0  # Positive slope = escalation
    negativity_factor = avg_negativity
    repetition_factor = 0.3 if pattern_analysis['has_repetitive_pattern'] else 0.0
    intensity_factor = recent_negativity * 0.5
    
    escalation_score = min(1.0, (
        trend_factor * 0.4 +
        negativity_factor * 0.3 +
        repetition_factor +
        intensity_factor
    ))
    
    # Determine if escalating
    is_escalating = escalation_score >= 0.5 or trend == "ESCALATING"
    
    # Collect indicators
    indicators = []
    if trend == "ESCALATING":
        indicators.append("Increasing negative sentiment")
    if avg_negativity > 0.6:
        indicators.append("High average negativity")
    if recent_negativity > 0.7:
        indicators.append("Recent message highly negative")
    if pattern_analysis['has_repetitive_pattern']:
        indicators.append("Repetitive negative language")
    if change_rate > 0.2:
        indicators.append("Rapid sentiment deterioration")
    
    # Determine severity
    severity = get_escalation_severity(escalation_score)
    
    return {
        "is_escalating": is_escalating,
        "escalation_score": escalation_score,
        "severity": severity,
        "trend": trend,
        "indicators": indicators,
        "sentiment_trajectory": negativity_scores,
        "average_negativity": avg_negativity,
        "change_rate": change_rate,
        "pattern_analysis": pattern_analysis
    }


def get_escalation_severity(escalation_score: float) -> str:
    """
    Convert escalation score to severity level
    
    Args:
        escalation_score: Score from 0.0 to 1.0
        
    Returns:
        Severity level string
    """
    if escalation_score >= 0.8:
        return "CRITICAL"
    elif escalation_score >= 0.6:
        return "HIGH"
    elif escalation_score >= 0.4:
        return "MODERATE"
    elif escalation_score >= 0.2:
        return "LOW"
    else:
        return "NONE"


def detect_rapid_fire_messages(timestamps: List[float], threshold_seconds: float = 5.0) -> bool:
    """
    Detect rapid-fire messaging pattern (potential escalation indicator)
    
    Args:
        timestamps: List of message timestamps
        threshold_seconds: Time window for rapid-fire detection
        
    Returns:
        Boolean indicating rapid-fire pattern detected
    """
    if len(timestamps) < 3:
        return False
    
    # Check if recent messages are within threshold
    recent_timestamps = timestamps[-3:]
    time_diffs = [
        recent_timestamps[i+1] - recent_timestamps[i]
        for i in range(len(recent_timestamps)-1)
    ]
    
    return all(diff <= threshold_seconds for diff in time_diffs)


def predict_escalation_risk(conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Predict future escalation risk based on conversation history
    
    Args:
        conversation_history: List of message dictionaries with 'text' and optional 'timestamp'
        
    Returns:
        Dictionary with risk prediction
    """
    messages = [msg['text'] for msg in conversation_history]
    
    # Get current escalation state
    current_state = detect_escalation(messages)
    
    # Check message frequency if timestamps available
    rapid_fire = False
    if all('timestamp' in msg for msg in conversation_history):
        timestamps = [msg['timestamp'] for msg in conversation_history]
        rapid_fire = detect_rapid_fire_messages(timestamps)
    
    # Calculate risk score
    base_risk = current_state['escalation_score']
    rapid_fire_bonus = 0.2 if rapid_fire else 0.0
    
    risk_score = min(1.0, base_risk + rapid_fire_bonus)
    
    # Determine risk level
    if risk_score >= 0.7:
        risk_level = "HIGH"
        recommendation = "Immediate intervention recommended"
    elif risk_score >= 0.5:
        risk_level = "MODERATE"
        recommendation = "Monitor closely, consider intervention"
    elif risk_score >= 0.3:
        risk_level = "LOW"
        recommendation = "Continue monitoring"
    else:
        risk_level = "MINIMAL"
        recommendation = "No action needed"
    
    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "recommendation": recommendation,
        "current_escalation": current_state,
        "rapid_fire_detected": rapid_fire
    }


if __name__ == "__main__":
    # Test the escalation detection
    test_conversation = [
        "Hey, how's it going?",
        "I'm a bit annoyed with the situation",
        "This is really getting on my nerves",
        "I'm getting really frustrated now!",
        "I'm so angry, this is unacceptable!"
    ]
    
    print("Testing Escalation Detection Model\n" + "="*50)
    
    # Test incremental escalation
    for i in range(2, len(test_conversation) + 1):
        messages = test_conversation[:i]
        result = detect_escalation(messages)
        print(f"\nMessages 1-{i}:")
        print(f"Is Escalating: {result['is_escalating']}")
        print(f"Escalation Score: {result['escalation_score']:.4f}")
        print(f"Severity: {result['severity']}")
        print(f"Trend: {result['trend']}")
        print(f"Indicators: {result['indicators']}")
