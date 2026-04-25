# ReproAgent Quick Start Script for Windows
# Run with: .\run.ps1

# Enable strict mode
$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "🚀 ReproAgent Quick Start (Windows)" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  ❌ Python not found! Please install Python 3.10+" -ForegroundColor Red
    exit 1
}

# Check if virtual environment exists
if (-Not (Test-Path "venv")) {
    Write-Host ""
    Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "  ✅ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host ""
Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1
Write-Host "  ✅ Activated" -ForegroundColor Green

# Install dependencies
Write-Host ""
Write-Host "📥 Installing dependencies..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt --quiet
Write-Host "  ✅ Dependencies installed" -ForegroundColor Green

# Create .env if not exists
if (-Not (Test-Path ".env")) {
    Write-Host ""
    Write-Host "📝 Creating .env file..." -ForegroundColor Yellow
    if (Test-Path ".env.example") {
        Copy-Item .env.example .env
    } else {
        "# Add your API keys here" | Out-File -FilePath .env -Encoding UTF8
    }
    Write-Host "  ⚠️  Please edit .env and add your API keys" -ForegroundColor Yellow
    Write-Host "  (Optional - system works without LLM)" -ForegroundColor Gray
}

# Create data directories
Write-Host ""
Write-Host "📁 Setting up data directories..." -ForegroundColor Yellow
$dirs = @(
    "data\papers\easy",
    "data\papers\medium", 
    "data\papers\hard",
    "logs",
    "checkpoints"
)
foreach ($dir in $dirs) {
    if (-Not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
Write-Host "  ✅ Directories created" -ForegroundColor Green

# Create sample data
Write-Host ""
Write-Host "📄 Creating sample papers..." -ForegroundColor Yellow
try {
    python -c "from reproagent.papers import create_sample_papers; create_sample_papers()" 2>$null
    Write-Host "  ✅ Sample data ready" -ForegroundColor Green
} catch {
    Write-Host "  ⚠️  Sample paper creation skipped" -ForegroundColor Yellow
}

# Validate environment
Write-Host ""
Write-Host "🔍 Validating environment..." -ForegroundColor Yellow
$validationResult = python validate.py
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Validation passed!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "⚠️  Some validations failed (may be non-critical)" -ForegroundColor Yellow
}

# Ask what to do
Write-Host ""
Write-Host ("=" * 50)
Write-Host "What would you like to do?"
Write-Host ("=" * 50)
Write-Host "1) Launch Gradio demo (recommended)"
Write-Host "2) Run inference"
Write-Host "3) Run baseline comparison"
Write-Host "4) Run validation only"
Write-Host "5) Exit"
Write-Host ""
$choice = Read-Host "Enter choice [1-5]"

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "🎨 Launching Gradio demo..." -ForegroundColor Cyan
        python server/app.py
    }
    "2" {
        Write-Host ""
        Write-Host "🤖 Running inference..." -ForegroundColor Cyan
        python inference.py --difficulty easy --steps 30
    }
    "3" {
        Write-Host ""
        Write-Host "📊 Running baseline comparison..." -ForegroundColor Cyan
        python baseline/run_baseline.py
    }
    "4" {
        Write-Host ""
        Write-Host "✅ Validation complete (already ran above)" -ForegroundColor Green
    }
    "5" {
        Write-Host "👋 Goodbye!" -ForegroundColor Cyan
        exit 0
    }
    default {
        Write-Host "Invalid choice. Exiting." -ForegroundColor Red
        exit 1
    }
}
