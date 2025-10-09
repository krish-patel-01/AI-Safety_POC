"""
Content Filtering Model
Age-appropriate content filtering for guardian-supervised accounts
"""
from transformers import pipeline
from typing import Dict, Any, List
import re
import warnings
warnings.filterwarnings('ignore')


# Global model cache
_content_classifier = None


# Content categories with age restrictions
AGE_RESTRICTIONS = {
    "violence": {"min_age": 13, "severity_threshold": 0.6},
    "sexual_content": {"min_age": 18, "severity_threshold": 0.5},
    "profanity": {"min_age": 13, "severity_threshold": 0.7},
    "drugs_alcohol": {"min_age": 16, "severity_threshold": 0.6},
    "horror_disturbing": {"min_age": 13, "severity_threshold": 0.7}
}


# Keyword patterns for different content categories
CONTENT_KEYWORDS = {
    "violence": [
        "kill", "murder", "stab", "shoot", "gun", "weapon", "fight", "assault",
        "attack", "violence", "blood", "gore", "brutal", "violent"
    ],
    "sexual_content": [
        "sex", "sexual", "explicit", "nude", "naked", "porn", "adult",
        "intimate", "erotic"
    ],
    "profanity": [
        "damn", "hell", "shit", "fuck", "bitch", "ass", "bastard", "crap"
    ],
    "drugs_alcohol": [
        "drug", "cocaine", "marijuana", "weed", "alcohol", "beer", "drunk",
        "high", "smoking", "cigarette", "vape"
    ],
    "horror_disturbing": [
        "scary", "horror", "terrifying", "nightmare", "creepy", "disturbing",
        "haunted", "demon", "ghost", "monster"
    ]
}


def load_content_filtering_model():
    """
    Load pretrained zero-shot classification model for content filtering
    
    Returns:
        Classification pipeline
    """
    global _content_classifier
    
    if _content_classifier is None:
        print("Loading content filtering model (zero-shot classification)...")
        
        # Use zero-shot classification for flexible content categorization
        _content_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        
        print("✓ Content filtering model loaded successfully")
    
    return _content_classifier


def detect_keywords(text: str) -> Dict[str, List[str]]:
    """
    Detect content keywords in text
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary mapping categories to found keywords
    """
    text_lower = text.lower()
    
    detected = {}
    for category, keywords in CONTENT_KEYWORDS.items():
        found = [kw for kw in keywords if kw in text_lower]
        if found:
            detected[category] = found
    
    return detected


def classify_content_category(text: str) -> Dict[str, float]:
    """
    Classify content into safety categories using zero-shot classification
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary mapping categories to confidence scores
    """
    classifier = load_content_filtering_model()
    
    # Defined candidate labels
    candidate_labels = [
        "violent content",
        "sexual content",
        "profane language",
        "drug or alcohol reference",
        "horror or disturbing content",
        "safe content"
    ]
    
    # Classify
    result = classifier(text, candidate_labels, multi_label=True)
    
    # Map to our categories
    category_mapping = {
        "violent content": "violence",
        "sexual content": "sexual_content",
        "profane language": "profanity",
        "drug or alcohol reference": "drugs_alcohol",
        "horror or disturbing content": "horror_disturbing",
        "safe content": "safe"
    }
    
    scores = {}
    for label, score in zip(result['labels'], result['scores']):
        mapped_category = category_mapping.get(label, label)
        scores[mapped_category] = score
    
    return scores


def filter_content(text: str, user_age: int = 13) -> Dict[str, Any]:
    """
    Filter content based on age appropriateness
    
    Args:
        text: Input text to analyze
        user_age: Age of user (default: 13)
        
    Returns:
        Dictionary containing:
            - is_appropriate: Boolean indicating if content is age-appropriate
            - flagged_categories: List of inappropriate content categories
            - category_scores: Confidence scores for each category
            - risk_score: Overall content risk score
            - recommendations: List of actions to take
    """
    # Keyword detection
    detected_keywords = detect_keywords(text)
    
    # ML-based classification
    category_scores = classify_content_category(text)
    
    # Combine keyword and ML signals
    flagged_categories = []
    risk_scores = []
    
    for category, restrictions in AGE_RESTRICTIONS.items():
        min_age = restrictions["min_age"]
        threshold = restrictions["severity_threshold"]
        
        # Get ML score
        ml_score = category_scores.get(category, 0.0)
        
        # Boost score if keywords detected
        keyword_boost = 0.2 if category in detected_keywords else 0.0
        combined_score = min(1.0, ml_score + keyword_boost)
        
        # Check if inappropriate for user age
        if user_age < min_age and combined_score >= threshold:
            flagged_categories.append({
                "category": category,
                "score": combined_score,
                "min_age_required": min_age,
                "keywords_found": detected_keywords.get(category, [])
            })
            risk_scores.append(combined_score)
    
    # Calculate overall risk
    overall_risk = max(risk_scores) if risk_scores else 0.0
    
    # Determine if appropriate
    is_appropriate = len(flagged_categories) == 0
    
    # Generate recommendations
    recommendations = []
    if not is_appropriate:
        recommendations.append("Content blocked - inappropriate for user age")
        recommendations.append(f"Requires guardian approval for user age {user_age}")
        
        for flag in flagged_categories:
            recommendations.append(
                f"Contains {flag['category']} (min age: {flag['min_age_required']})"
            )
    
    return {
        "is_appropriate": is_appropriate,
        "flagged_categories": flagged_categories,
        "category_scores": category_scores,
        "risk_score": overall_risk,
        "recommendations": recommendations,
        "user_age": user_age,
        "detected_keywords": detected_keywords
    }


def get_content_rating(category_scores: Dict[str, float]) -> str:
    """
    Determine content rating based on category scores
    
    Args:
        category_scores: Dictionary of category scores
        
    Returns:
        Content rating string (G, PG, PG-13, R, NC-17)
    """
    max_score = 0.0
    flagged_category = None
    
    for category, score in category_scores.items():
        if category != "safe" and score > max_score:
            max_score = score
            flagged_category = category
    
    if max_score < 0.3:
        return "G"  # General audiences
    elif max_score < 0.5:
        return "PG"  # Parental guidance suggested
    elif max_score < 0.7:
        return "PG-13"  # Parents strongly cautioned
    elif max_score < 0.85:
        return "R"  # Restricted
    else:
        return "NC-17"  # Adults only


def create_guardian_report(filtering_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create summary report for guardians
    
    Args:
        filtering_results: List of filtering results
        
    Returns:
        Summary report dictionary
    """
    total_messages = len(filtering_results)
    flagged_messages = sum(1 for r in filtering_results if not r['is_appropriate'])
    
    # Aggregate categories
    category_counts = {}
    for result in filtering_results:
        for flag in result.get('flagged_categories', []):
            category = flag['category']
            category_counts[category] = category_counts.get(category, 0) + 1
    
    # Calculate risk level
    if flagged_messages == 0:
        risk_level = "LOW"
    elif flagged_messages / total_messages < 0.2:
        risk_level = "MODERATE"
    else:
        risk_level = "HIGH"
    
    return {
        "total_messages": total_messages,
        "flagged_messages": flagged_messages,
        "flag_rate": flagged_messages / total_messages if total_messages > 0 else 0.0,
        "category_counts": category_counts,
        "risk_level": risk_level,
        "requires_guardian_review": flagged_messages > 0
    }


if __name__ == "__main__":
    # Test the content filtering
    test_messages = [
        "Let's go to the park and play!",
        "That movie has a lot of violence and blood",
        "Want to grab a beer tonight?",
        "This horror film is really scary and disturbing",
        "Hey, what the hell are you doing?"
    ]
    
    test_ages = [8, 13, 16]
    
    print("Testing Content Filtering Model\n" + "="*50)
    
    for age in test_ages:
        print(f"\n{'='*50}")
        print(f"Testing for age: {age}")
        print('='*50)
        
        for msg in test_messages[:3]:  # Test first 3 messages
            result = filter_content(msg, user_age=age)
            print(f"\nMessage: {msg}")
            print(f"Is Appropriate: {result['is_appropriate']}")
            print(f"Risk Score: {result['risk_score']:.4f}")
            print(f"Content Rating: {get_content_rating(result['category_scores'])}")
            if result['flagged_categories']:
                print(f"Flagged: {[f['category'] for f in result['flagged_categories']]}")
