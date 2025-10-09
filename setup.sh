#!/bin/bash
# Quick setup script for Unix/Linux/Mac

echo "================================================"
echo "AI Safety Models POC - Quick Setup"
echo "================================================"
echo ""

echo "[1/3] Checking Python installation..."
python3 --version
if [ $? -ne 0 ]; then
    echo "ERROR: Python not found! Please install Python 3.8+"
    exit 1
fi
echo ""

echo "[2/3] Installing dependencies..."
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi
echo ""

echo "[3/3] Setup complete!"
echo ""
echo "================================================"
echo "Next Steps:"
echo "================================================"
echo ""
echo "Option 1: Run CLI Demo"
echo "  python3 cli_chat.py"
echo ""
echo "Option 2: Run Evaluation"
echo "  python3 evaluate.py"
echo ""
echo "Option 3: Test Individual Models"
echo "  python3 models/abuse_detection.py"
echo "  python3 models/crisis_detection.py"
echo "  python3 models/escalation_detection.py"
echo "  python3 models/content_filtering.py"
echo ""
echo "================================================"
echo ""
