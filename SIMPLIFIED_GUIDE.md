# AI Safety Models POC - Simplified Version

## What Changed

I simplified the codebase to be a proper **POC** (Proof of Concept) rather than production-ready code.

## New Simple Files

### Core Files (Use These!)

1. **models_orchestrator.py** - Main logic
   - Simple function: `analyze_message(text)`
   - Runs all 4 models
   - Returns combined results

2. **demo_cli.py** - Interactive demo
   ```bash
   python demo_cli.py
   ```
   - Option 1: Demo mode (predefined messages)
   - Option 2: Interactive (type your own)

3. **evaluate.py** - Basic testing
   ```bash
   python evaluate.py
   ```
   - Tests on 4 sample cases
   - Shows accuracy

4. **README_SIMPLE.md** - Simplified documentation
   - Quick start guide
   - No production complexity

### Model Files (Unchanged)

The 4 model files are still the same:
- `models/abuse_detection.py`
- `models/crisis_detection.py`
- `models/escalation_detection.py`
- `models/content_filtering.py`

## How to Use this POC

```bash
# 1. Install
pip install -r requirements.txt

# 2. Run demo
python demo_cli.py

# Select option 1, press Enter to see each message analyzed
```


## Files to Focus On

**For Demo:**
- `demo_cli.py` - Run this
- `models_orchestrator.py` - Show this code
- `README.md` - Explain from thi

**Ignore:**
- `orchestrator.py` (old complex version)
- `cli_chat.py` (old complex version)
- `evaluate.py` (old complex version)
- Long documentation files

## Summary

**Use the SIMPLE files:**
- `models_orchestrator.py`
- `demo_cli.py`
- `evaluate.py`
- `README.md`

**They demonstrate:**
✅ All 4 models working  
✅ Function-based approach  
✅ Pretrained models  
✅ Real-time analysis  
✅ Simple, understandable POC  

**This is what a POC should look like!**
