# AI Safety Models — Proof of Concept

Four pretrained transformer models composed into a single chat-safety check. Each
message is scored independently for **abuse**, **crisis signals**, **conversational
escalation** and **age-appropriateness**; a message is reported safe only if all four
agree.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

> **Proof of concept, not a safety system.** Thresholds are hand-picked rather than
> tuned, the evaluation set is four messages, and no model here has been validated on
> real user data. See [Limitations](#limitations) before drawing conclusions from it.

---

## The four detectors

| Detector | Model | Reads | Flags when |
|---|---|---|---|
| Abuse | [`unitary/toxic-bert`](https://huggingface.co/unitary/toxic-bert) | current message | any toxicity category crosses threshold |
| Crisis | [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) | current message | distress emotions + crisis phrase patterns |
| Escalation | [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) | conversation history | sentiment trends negative across messages |
| Content filter | [`facebook/bart-large-mnli`](https://huggingface.co/facebook/bart-large-mnli) | current message + user age | zero-shot match on age-inappropriate categories |

Escalation is the only one that needs history — the other three are pure functions of a
single message. That's why `analyze_message` takes a `conversation_history` list.

```
                       ┌──────────────────────────────┐
   message ───────────▶│      analyze_message()       │
   history ───────────▶│    models_orchestrator.py    │
   user_age ──────────▶└──────────────┬───────────────┘
                                      │ fan-out
              ┌───────────┬───────────┼───────────┬────────────┐
              ▼           ▼           ▼           ▼            │
          ┌───────┐  ┌────────┐  ┌──────────┐ ┌─────────┐      │
          │ abuse │  │ crisis │  │escalation│ │ content │      │
          └───┬───┘  └───┬────┘  └────┬─────┘ └────┬────┘      │
              └──────────┴────────────┴────────────┘           │
                                      │ is_safe = NOT any flag │
                                      ▼                        │
                            { is_safe, per-model detail } ◀────┘
```

`is_safe` is a plain OR over the four flags — any single detector firing marks the
message unsafe. There is no weighting or confidence combination.

## Quick start

```bash
git clone https://github.com/krish-patel-01/AI-Safety_POC.git
cd AI-Safety_POC

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python demo_cli.py
```

`setup.sh` (Linux/macOS) and `setup.bat` (Windows) do the same thing.

The first run downloads roughly **2 GB** of model weights from Hugging Face and takes
1–2 minutes. They're cached in `~/.cache/huggingface` afterwards, so later runs start
in seconds.

## Usage

### Interactive demo

```bash
python demo_cli.py
```

Two modes: option `1` walks through seven predefined messages that each trip a
different detector; option `2` lets you type your own. Models are loaded once via
`warmup_models()` before either mode starts, rather than on every message.

### As a library

```python
from models_orchestrator import analyze_message, warmup_models

warmup_models()   # optional, but avoids a slow first call

result = analyze_message(
    "I can't take this anymore",
    conversation_history=["hey", "this is annoying"],
    user_age=13,
)

result["is_safe"]                            # False
result["crisis_detection"]["risk_score"]     # 0.0–1.0
result["abuse_detection"]["flagged_categories"]
```

`analyze_message` returns a dict with `message`, `is_safe`, and one sub-dict per
detector, each carrying its own score and flags. Nothing is logged or persisted.

### Evaluation

```bash
python evaluate.py
```

Runs four labelled cases and prints accuracy. This is a smoke test that the pipeline
produces sane output — it is not a benchmark, and the number it prints should not be
quoted as one.

## Configuration

Thresholds live as defaults in each detector's function signature:

| Detector | Parameter | Default |
|---|---|---|
| `detect_abuse` | `threshold` | toxicity score above which a category is flagged |
| `detect_crisis` | `threshold` | distress score required to flag crisis |
| `detect_escalation` | `window_size` | how many recent messages form the trend |
| `filter_content` | `user_age` | drives which categories count as inappropriate |

They were chosen by hand on the demo messages. Any real deployment needs them
re-tuned against its own labelled data.

## Project layout

```
├── models_orchestrator.py   # fan-out, is_safe aggregation, result formatting
├── demo_cli.py              # interactive CLI (demo + freeform modes)
├── evaluate.py              # 4-case smoke test
├── models/
│   ├── abuse_detection.py       # toxic-bert
│   ├── crisis_detection.py      # emotion-distilroberta + phrase patterns
│   ├── escalation_detection.py  # twitter-roberta sentiment trend
│   └── content_filtering.py     # bart-large-mnli zero-shot
├── ARCHITECTURE.md          # design notes
├── TECHNICAL_REPORT.md      # model selection and rationale
└── docs/problem-statement.txt   # original brief
```

Each detector module exposes a `load_*_model()` that memoises the model in a module
global, and a `detect_*`/`filter_*` function that uses it.

## Limitations

These matter more than the feature list:

- **English only.** Every model is English-trained; other languages will score
  arbitrarily rather than fail loudly.
- **Thresholds are guesses.** Not tuned, not validated, no measured false-positive or
  false-negative rate.
- **The evaluation set is four messages.** It cannot support any accuracy claim.
- **Crisis detection is not a safety net.** It combines an emotion classifier with
  keyword patterns and will miss indirect or coded expressions of distress. Do not put
  it in a path where a missed detection harms someone.
- **No adversarial robustness.** Obfuscation, character substitution and paraphrase
  defeat these classifiers easily.
- **Sequential inference on CPU.** Four models run one after another per message —
  usable for a demo, too slow for live chat at volume.
- **Known biases.** Toxicity classifiers over-flag African-American English and text
  discussing identity terms. `unitary/toxic-bert` inherits this from its training data.

If you're building something real for crisis response, route to trained humans and
established services, not to this.

## Development

```bash
pip install -e ".[dev]"
ruff check .
```

## License

[MIT](LICENSE)
