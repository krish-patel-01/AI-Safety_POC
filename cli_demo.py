"""
Simple CLI Demo - POC
Interactive demo of AI Safety Models
"""
from models_orchestrator import analyze_message, print_results, warmup_models


def run_demo():
    """Run a simple demo with predefined messages"""
    print("\n" + "="*70)
    print("AI SAFETY MODELS POC - DEMO")
    print("="*70)
    print("\nDemonstrating all 4 models:")
    print("  1. Abuse Language Detection")
    print("  2. Crisis Intervention")
    print("  3. Escalation Pattern Recognition")
    print("  4. Content Filtering")
    print("\n" + "="*70 + "\n")
    
    demo_messages = [
        "Hi! How are you today?",
        "This is really annoying me",
        "You're such an idiot!",
        "I can't take this anymore",
        "I want to end it all, there's no point",
        "Want to watch that horror movie?",
        "fuck you all"
    ]
    
    conversation_history = []
    
    for i, msg in enumerate(demo_messages, 1):
        print(f"\n>>> Message {i}/{len(demo_messages)}")
        input("Press Enter to analyze next message...")
        
        result = analyze_message(msg, conversation_history.copy(), user_age=13)
        print_results(result)
        conversation_history.append(msg)
    
    print("\n✓ Demo complete!")


def run_interactive():
    """Interactive mode - type your own messages"""
    print("\n" + "="*70)
    print("AI SAFETY MODELS POC - INTERACTIVE MODE")
    print("="*70)
    print("\nType messages to analyze (type 'quit' to exit)\n")
    
    conversation_history = []
    
    while True:
        message = input("You: ").strip()
        
        if not message:
            continue
        
        if message.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        result = analyze_message(message, conversation_history.copy(), user_age=13)
        print_results(result)
        conversation_history.append(message)


def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("           AI SAFETY MODELS - PROOF OF CONCEPT")
    print("="*70)
    print("\nSelect mode:")
    print("  1. Run Demo (predefined messages)")
    print("  2. Interactive Mode (type your own)")
    print("  3. Exit")
    print("\n" + "="*70)
    
    choice = input("\nChoice (1-3): ").strip()
    
    if choice == '1':
        warmup_models()  # Load models once before demo
        run_demo()
    elif choice == '2':
        warmup_models()  # Load models once before interactive
        run_interactive()
    elif choice == '3':
        print("\nGoodbye!")
    else:
        print("\nInvalid choice!")


if __name__ == "__main__":
    main()
