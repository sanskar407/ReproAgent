@echo off
setlocal enabledelayedexpansion

echo.
echo 🚀 ReproAgent Quick Start (Windows)
echo ====================================
echo.

:: Check Python
echo Checking Python version...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   ❌ Python not found! Install Python 3.10+
    exit /b 1
)
python --version
echo.

:: Create venv if needed
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
    echo   ✅ Virtual environment created
    echo.
)

:: Activate venv
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat
echo   ✅ Activated
echo.

:: Install dependencies
echo 📥 Installing dependencies...
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt --quiet
echo   ✅ Dependencies installed
echo.

:: Create .env
if not exist ".env" (
    echo 📝 Creating .env file...
    if exist ".env.example" (
        copy .env.example .env >nul
    ) else (
        echo # Add your API keys here > .env
    )
    echo   ⚠️  Edit .env to add API keys (optional)
    echo.
)

:: Create directories
echo 📁 Setting up directories...
mkdir data\papers\easy 2>nul
mkdir data\papers\medium 2>nul
mkdir data\papers\hard 2>nul
mkdir logs 2>nul
mkdir checkpoints 2>nul
echo   ✅ Directories created
echo.

:: Create sample data
echo 📄 Creating sample papers...
python -c "from reproagent.papers import create_sample_papers; create_sample_papers()" 2>nul
if %errorlevel% equ 0 (
    echo   ✅ Sample data ready
) else (
    echo   ⚠️  Sample paper creation skipped
)
echo.

:: Validate
echo 🔍 Validating environment...
python validate.py
echo.

:: Menu
echo ==================================================
echo What would you like to do?
echo ==================================================
echo 1^) Launch Gradio demo ^(recommended^)
echo 2^) Run inference
echo 3^) Run baseline comparison
echo 4^) Run validation only
echo 5^) Exit
echo.
set /p choice="Enter choice [1-5]: "

if "%choice%"=="1" (
    echo.
    echo 🎨 Launching Gradio demo...
    python server/app.py
) else if "%choice%"=="2" (
    echo.
    echo 🤖 Running inference...
    python inference.py --difficulty easy --steps 30
) else if "%choice%"=="3" (
    echo.
    echo 📊 Running baseline comparison...
    python baseline/run_baseline.py
) else if "%choice%"=="4" (
    echo.
    echo ✅ Validation complete
) else if "%choice%"=="5" (
    echo 👋 Goodbye!
    exit /b 0
) else (
    echo Invalid choice.
    exit /b 1
)
