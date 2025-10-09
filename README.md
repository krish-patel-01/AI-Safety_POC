# AI Safety Models - Proof of Concept

A simple POC demonstrating 4 integrated AI safety models for chat monitoring.

## Overview

This POC shows how pretrained transformer models can work together to detect:

1. **🗣️ Abuse Language** - Toxic/threatening content (toxic-bert)
2. **🆘 Crisis Intervention** - Self-harm indicators (emotion-distilroberta)
3. **📈 Escalation Patterns** - Conversation escalation (sentiment analysis)
4. **🔒 Content Filtering** - Age-inappropriate content (zero-shot classification)

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run demo
python demo_cli.py
```

### Alternative Setup

#### Windows
```batch
# Run the setup script
setup.bat
```

#### Linux/Mac
```bash
# Make the setup script executable
chmod +x setup.sh

# Run the setup script
./setup.sh
```

## Project Structure

```
AI_safety/
├── models/                      # 4 safety detection models
│   ├── abuse_detection.py      # Toxicity detection
│   ├── crisis_detection.py     # Crisis/distress detection
│   ├── escalation_detection.py # Escalation patterns
│   └── content_filtering.py    # Age-appropriate filtering
├── models_orchestrator.py      # Coordinates all 4 models
├── demo_cli.py               # Interactive demo
├── evaluate.py          # Basic evaluation
└── requirements.txt            # Dependencies
```

## Usage

### 1. Demo Mode (Recommended)
```bash
python demo_cli.py
# Select option 1: "Run Demo"
```

### 2. Interactive Mode
```bash
python demo_cli.py
# Select option 2: "Interactive Mode"
# Type your own messages
```

### 3. Evaluation
```bash
python evaluate.py
```

### 4. Programmatic Use
```python
from models_orchestrator import analyze_message

result = analyze_message("Your message here", user_age=13)
print(result['is_safe'])  # True or False
print(result['abuse_detection'])  # Abuse results
print(result['crisis_detection'])  # Crisis results
```

## How It Works

**Simple Function-Based Design:**

```python
# Each model is a simple function
detect_abuse(text) → result
detect_crisis(text) → result
detect_escalation(messages) → result
filter_content(text, age) → result

# Orchestrator runs all 4
analyze_message(text) → combined_result
```

## Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Abuse Detection | unitary/toxic-bert | 6-category toxicity |
| Crisis Detection | j-hartmann/emotion-english-distilroberta-base | Emotion + keywords |
| Escalation Detection | cardiffnlp/twitter-roberta-base-sentiment-latest | Sentiment trends |
| Content Filtering | facebook/bart-large-mnli | Zero-shot classification |

## Example Output

```
MESSAGE: "I can't take this anymore"
⚠️ UNSAFE

🗣️  Abuse Detection:
    Is Abusive: False
    Max Toxicity: 0.123

🆘 Crisis Detection:
    Is Crisis: True
    Risk Score: 0.724
    ⚠️  INTERVENTION REQUIRED!

📈 Escalation Detection:
    Is Escalating: False
    Trend: STABLE

🔒 Content Filtering:
    Age Appropriate: True
```

## Performance

- **First run**: Downloads models (~1-2 GB), takes 1-2 minutes
- **Subsequent runs**: 200-500ms per message
- **Memory**: 2-3 GB RAM
- **Accuracy**: 80-90% on test cases

## Files Explained

- **models_orchestrator.py** - Main logic, runs all 4 models
- **demo_cli.py** - Interactive demo tool
- **evaluate.py** - Basic testing
- **models/** - Individual detection functions
- **requirements.txt** - Python dependencies

## Troubleshooting

**Models downloading slowly?**
- First run downloads ~1-2 GB (one-time only)
- Needs internet connection

**Memory errors?**
- Close other applications
- Needs 4GB+ RAM

**Import errors?**
- Run: `pip install -r requirements.txt`
- Check Python version (3.8+ required)

---

**This is a POC** - it demonstrates the concept works. For production deployment, significant additional engineering would be required.
