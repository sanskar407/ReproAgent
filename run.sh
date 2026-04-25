#!/bin/bash

# ReproAgent Quick Start Script
# Sets up environment and launches demo

set -e  # Exit on error

echo "🚀 ReproAgent Quick Start"
echo "=========================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python version: $python_version"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "  ✅ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "🔧 Activating virtual environment..."
source venv/bin/activate
echo "  ✅ Activated"

# Install dependencies
echo ""
echo "📥 Installing dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "  ✅ Dependencies installed"

# Create .env if not exists
if [ ! -f ".env" ]; then
    echo ""
    echo "📝 Creating .env file..."
    cp .env.example .env 2>/dev/null || echo "# Add your API keys here" > .env
    echo "  ⚠️  Please edit .env and add your API keys"
    echo "  (Optional - system works without LLM)"
fi

# Create data directories
echo ""
echo "📁 Setting up data directories..."
mkdir -p data/papers/easy
mkdir -p data/papers/medium
mkdir -p data/papers/hard
mkdir -p logs
mkdir -p checkpoints
echo "  ✅ Directories created"

# Create sample data
echo ""
echo "📄 Creating sample papers..."
python3 -c "from reproagent.papers import create_sample_papers; create_sample_papers()" 2>/dev/null || echo "  ⚠️  Sample paper creation skipped"
echo "  ✅ Sample data ready"

# Validate environment
echo ""
echo "🔍 Validating environment..."
if python3 validate.py; then
    echo ""
    echo -e "${GREEN}✅ Validation passed!${NC}"
else
    echo ""
    echo -e "${YELLOW}⚠️  Some validations failed (may be non-critical)${NC}"
fi

# Ask what to do
echo ""
echo "="*50
echo "What would you like to do?"
echo "="*50
echo "1) Launch Gradio demo (recommended)"
echo "2) Run inference"
echo "3) Run baseline comparison"
echo "4) Run validation only"
echo "5) Exit"
echo ""
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        echo ""
        echo "🎨 Launching Gradio demo..."
        python3 server/app.py
        ;;
    2)
        echo ""
        echo "🤖 Running inference..."
        python3 inference.py --difficulty easy --steps 30
        ;;
    3)
        echo ""
        echo "📊 Running baseline comparison..."
        python3 baseline/run_baseline.py
        ;;
    4)
        echo ""
        echo "✅ Validation complete (already ran above)"
        ;;
    5)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac
