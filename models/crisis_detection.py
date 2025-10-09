"""
Crisis Intervention Model
Detects severe emotional distress and self-harm indicators
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from typing import Dict, Any, List
import re
import warnings
warnings.filterwarnings('ignore')


# Global model cache
_crisis_model = None
_crisis_tokenizer = None
_emotion_classifier = None


# Crisis keywords and patterns
SELF_HARM_KEYWORDS = [
    "suicide", "kill myself", "end it all", "want to die", "better off dead",
    "can't go on", "no reason to live", "self harm", "hurt myself", "cut myself",
    "overdose", "end my life", "not worth living"
]

DISTRESS_KEYWORDS = [
    "depressed", "hopeless", "worthless", "give up", "can't take it",
    "falling apart", "no way out", "drowning", "suffocating", "trapped",
    "can't breathe", "breaking down", "losing control"
]


def load_crisis_detection_model():
    """
    Load pretrained models for crisis detection
    Uses emotion/distress classification models
    
    Returns:
        Tuple of (model, tokenizer, emotion_classifier)
    """
    global _crisis_model, _crisis_tokenizer, _emotion_classifier
    
    if _emotion_classifier is None:
        print("Loading crisis detection models...")
        
        # Used emotion classification model
        model_name = "j-hartmann/emotion-english-distilroberta-base"
        _crisis_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _crisis_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _crisis_model.eval()

        # Emotion classifier pipeline
        _emotion_classifier = pipeline(
            "text-classification",
            model=model_name,
            return_all_scores=True
        )
        
        print("✓ Crisis detection models loaded successfully")
    
    return _crisis_model, _crisis_tokenizer, _emotion_classifier


def check_crisis_keywords(text: str) -> Dict[str, Any]:
    """
    Check for crisis-related keywords in text
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary with keyword analysis results
    """
    text_lower = text.lower()
    
    # Check for self-harm indicators
    self_harm_found = [kw for kw in SELF_HARM_KEYWORDS if kw in text_lower]
    
    # Check for distress indicators
    distress_found = [kw for kw in DISTRESS_KEYWORDS if kw in text_lower]
    
    has_self_harm = len(self_harm_found) > 0
    has_distress = len(distress_found) > 0
    
    # Calculate keyword-based risk score
    keyword_risk = 0.0
    if has_self_harm:
        keyword_risk = 1.0  # Maximum risk for self-harm keywords
    elif has_distress:
        keyword_risk = 0.7  # High risk for distress keywords
    
    return {
        "has_self_harm_indicators": has_self_harm,
        "self_harm_keywords": self_harm_found,
        "has_distress_indicators": has_distress,
        "distress_keywords": distress_found,
        "keyword_risk_score": keyword_risk
    }


def analyze_emotional_state(text: str) -> Dict[str, Any]:
    """
    Analyze emotional state using pretrained emotion classifier
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary with emotion analysis results
    """
    _, _, emotion_classifier = load_crisis_detection_model()
    
    # Get emotion predictions
    emotions = emotion_classifier(text)[0]
    
    # Convert to dictionary
    emotion_scores = {
        emotion['label']: emotion['score']
        for emotion in emotions
    }
    
    # Identify dominant emotion
    dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
    
    # Calculate distress level based on negative emotions
    negative_emotions = ['sadness', 'fear', 'anger', 'disgust']
    distress_score = sum(
        emotion_scores.get(emotion, 0.0) 
        for emotion in negative_emotions
    ) / len(negative_emotions)
    
    return {
        "emotion_scores": emotion_scores,
        "dominant_emotion": dominant_emotion[0],
        "dominant_emotion_confidence": dominant_emotion[1],
        "distress_score": distress_score
    }


def detect_crisis(text: str, threshold: float = 0.6) -> Dict[str, Any]:
    """
    Comprehensive crisis detection combining keywords and emotion analysis
    
    Args:
        text: Input text to analyze
        threshold: Risk threshold for flagging (default: 0.6)
        
    Returns:
        Dictionary containing:
            - is_crisis: Boolean indicating if crisis detected
            - risk_score: Overall risk score (0.0 to 1.0)
            - severity: Crisis severity level
            - indicators: List of detected crisis indicators
            - requires_intervention: Boolean for immediate intervention need
            - emotion_analysis: Detailed emotion breakdown
            - keyword_analysis: Keyword detection results
    """
    # Keyword-based detection
    keyword_results = check_crisis_keywords(text)
    
    # Emotion-based detection
    emotion_results = analyze_emotional_state(text)
    
    # Combine scores (weighted average)
    keyword_weight = 0.6
    emotion_weight = 0.4
    
    risk_score = (
        keyword_results['keyword_risk_score'] * keyword_weight +
        emotion_results['distress_score'] * emotion_weight
    )
    
    # Determine if crisis
    is_crisis = risk_score >= threshold
    
    # Check if immediate intervention required (self-harm indicators)
    requires_intervention = keyword_results['has_self_harm_indicators']
    
    # Collect indicators
    indicators = []
    if keyword_results['has_self_harm_indicators']:
        indicators.append("Self-harm language detected")
    if keyword_results['has_distress_indicators']:
        indicators.append("Emotional distress detected")
    if emotion_results['distress_score'] > 0.7:
        indicators.append("High negative emotion levels")
    
    # Determine severity
    severity = get_crisis_severity(risk_score, requires_intervention)
    
    return {
        "is_crisis": is_crisis,
        "risk_score": risk_score,
        "severity": severity,
        "indicators": indicators,
        "requires_intervention": requires_intervention,
        "emotion_analysis": emotion_results,
        "keyword_analysis": keyword_results,
        "confidence": risk_score if is_crisis else (1.0 - risk_score)
    }


def get_crisis_severity(risk_score: float, has_self_harm: bool) -> str:
    """
    Determine crisis severity level
    
    Args:
        risk_score: Overall risk score
        has_self_harm: Whether self-harm indicators present
        
    Returns:
        Severity level string
    """
    if has_self_harm or risk_score >= 0.9:
        return "CRITICAL"
    elif risk_score >= 0.7:
        return "SEVERE"
    elif risk_score >= 0.5:
        return "MODERATE"
    elif risk_score >= 0.3:
        return "LOW"
    else:
        return "NONE"


def analyze_crisis_trajectory(messages: List[str], window_size: int = 10) -> Dict[str, Any]:
    """
    Analyze crisis risk trajectory across multiple messages
    
    Args:
        messages: List of message texts (chronological order)
        window_size: Number of recent messages to analyze
        
    Returns:
        Dictionary with trajectory analysis
    """
    recent_messages = messages[-window_size:] if len(messages) > window_size else messages
    
    results = [detect_crisis(msg) for msg in recent_messages]
    
    risk_scores = [r['risk_score'] for r in results]
    crisis_count = sum(1 for r in results if r['is_crisis'])
    
    # Check if risk is increasing
    is_worsening = False
    if len(risk_scores) >= 3:
        # Calculate trend
        recent_avg = sum(risk_scores[-3:]) / 3
        earlier_avg = sum(risk_scores[:-3]) / len(risk_scores[:-3]) if len(risk_scores) > 3 else 0
        is_worsening = recent_avg > earlier_avg + 0.1
    
    current_risk = risk_scores[-1] if risk_scores else 0.0
    
    return {
        "messages_analyzed": len(recent_messages),
        "crisis_messages_count": crisis_count,
        "current_risk_score": current_risk,
        "average_risk_score": sum(risk_scores) / len(risk_scores) if risk_scores else 0.0,
        "is_worsening": is_worsening,
        "trajectory": "WORSENING" if is_worsening else "STABLE",
        "requires_monitoring": current_risk > 0.4
    }


if __name__ == "__main__":
    # Test the crisis detection
    test_messages = [
        "I'm feeling a bit sad today",
        "I can't take this anymore, everything is falling apart",
        "I want to end it all, there's no point in going on",
        "Had a great day at the park with friends!",
        "I'm so depressed and hopeless, I don't see a way out"
    ]
    
    print("Testing Crisis Detection Model\n" + "="*50)
    for msg in test_messages:
        result = detect_crisis(msg)
        print(f"\nMessage: {msg}")
        print(f"Is Crisis: {result['is_crisis']}")
        print(f"Risk Score: {result['risk_score']:.4f}")
        print(f"Severity: {result['severity']}")
        print(f"Requires Intervention: {result['requires_intervention']}")
        print(f"Indicators: {result['indicators']}")
