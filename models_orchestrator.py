"""
Simple AI Safety Orchestrator - POC
Coordinates all 4 safety models
"""
from typing import Any

from models.abuse_detection import detect_abuse, load_abuse_detection_model
from models.content_filtering import filter_content, load_content_filtering_model
from models.crisis_detection import detect_crisis, load_crisis_detection_model
from models.escalation_detection import detect_escalation, load_escalation_detection_model


def warmup_models():
    """
    Preload all models once at startup to avoid reloading on each message
    """
    print("\n" + "="*70)
    print("Loading AI Safety Models...")
    print("="*70)
    print("\n⏳ This will take 1-2 minutes on first run (models are cached after)...\n")

    # Load all 4 models
    load_abuse_detection_model()
    load_crisis_detection_model()
    load_escalation_detection_model()
    load_content_filtering_model()

    print("\n✓ All models loaded and ready!\n")
    print("="*70 + "\n")


def analyze_message(message: str, conversation_history: list = None, user_age: int = 13) -> dict[str, Any]:
    """
    Simple function to analyze a message through all 4 safety models

    Args:
        message: Text to analyze
        conversation_history: List of previous messages for escalation detection
        user_age: User age for content filtering

    Returns:
        Dictionary with results from all 4 models
    """
    # Use history or just current message
    if conversation_history is None:
        conversation_history = []
    conversation_history.append(message)

    print(f"\nAnalyzing: '{message}'...")

    # Run all 4 models
    print("  - Running abuse detection...")
    abuse_result = detect_abuse(message)

    print("  - Running crisis detection...")
    crisis_result = detect_crisis(message)

    print("  - Running escalation detection...")
    escalation_result = detect_escalation(conversation_history)

    print("  - Running content filtering...")
    content_result = filter_content(message, user_age)

    # Safety check
    is_safe = not (
        abuse_result.get("is_abusive", False) or
        crisis_result.get("is_crisis", False) or
        escalation_result.get("is_escalating", False) or
        not content_result.get("is_appropriate", True)
    )

    print(f"  ✓ Analysis complete! Safe: {is_safe}\n")

    return {
        "message": message,
        "is_safe": is_safe,
        "abuse_detection": abuse_result,
        "crisis_detection": crisis_result,
        "escalation_detection": escalation_result,
        "content_filtering": content_result
    }


def print_results(result: dict[str, Any]):
    """Print analysis results in a readable format"""
    print("="*70)
    print(f"MESSAGE: {result['message']}")
    print("="*70)
    print(f"\n{'SAFE' if result['is_safe'] else '⚠️ UNSAFE'}\n")

    # Abuse Detection
    abuse = result['abuse_detection']
    print("🗣️  Abuse Detection:")
    print(f"    Is Abusive: {abuse.get('is_abusive', False)}")
    print(f"    Max Toxicity: {abuse.get('max_toxicity', 0):.3f}")
    if abuse.get('flagged_categories'):
        print(f"    Categories: {', '.join(abuse['flagged_categories'])}")

    # Crisis Detection
    crisis = result['crisis_detection']
    print("\n🆘 Crisis Detection:")
    print(f"    Is Crisis: {crisis.get('is_crisis', False)}")
    print(f"    Risk Score: {crisis.get('risk_score', 0):.3f}")
    if crisis.get('requires_intervention'):
        print("    ⚠️  INTERVENTION REQUIRED!")

    # Escalation Detection
    escalation = result['escalation_detection']
    print("\n📈 Escalation Detection:")
    print(f"    Is Escalating: {escalation.get('is_escalating', False)}")
    print(f"    Score: {escalation.get('escalation_score', 0):.3f}")
    print(f"    Trend: {escalation.get('trend', 'N/A')}")

    # Content Filtering
    content = result['content_filtering']
    print("\n🔒 Content Filtering:")
    print(f"    Age Appropriate: {content.get('is_appropriate', True)}")
    print(f"    Risk Score: {content.get('risk_score', 0):.3f}")
    if content.get('flagged_categories'):
        print(f"    Flagged: {len(content['flagged_categories'])} categories")

    print("\n" + "="*70 + "\n")


# Simple demo function
if __name__ == "__main__":
    print("\n" + "="*70)
    print("AI SAFETY MODELS POC - SIMPLE DEMO")
    print("="*70)

    # Test messages
    test_messages = [
        "Hello! How are you doing today?",
        "You're such an idiot!",
        "I can't take this anymore, I want to end it all",
        "I'm getting really frustrated now!",
        "Want to watch that horror movie with violence?"
    ]

    conversation_history = []

    for msg in test_messages:
        result = analyze_message(msg, conversation_history.copy())
        print_results(result)
        conversation_history.append(msg)
