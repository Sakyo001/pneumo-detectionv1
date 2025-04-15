@echo off
echo =========================================
echo  Model Diagnostics Tool
echo =========================================

REM Get the current directory
SET DIR=%~dp0
SET MODEL_DIR=%DIR%\..\output

echo.
echo Checking Python and PyTorch...
python --version
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torchvision; print(f'Torchvision version: {torchvision.__version__}')"

REM List model files
echo.
echo Checking model files...
dir "%MODEL_DIR%"

REM Run detailed analysis on final_model.pth
SET FINAL_MODEL=%MODEL_DIR%\final_model.pth
if exist "%FINAL_MODEL%" (
    echo.
    echo Analyzing final_model.pth...
    python "%DIR%\model_debug.py" --model "%FINAL_MODEL%"
)

REM Run detailed analysis on best_model.pth
SET BEST_MODEL=%MODEL_DIR%\best_model.pth
if exist "%BEST_MODEL%" (
    echo.
    echo Analyzing best_model.pth...
    python "%DIR%\model_debug.py" --model "%BEST_MODEL%"
)

REM Test prediction with a sample image
echo.
echo Running sample prediction test...
SET SAMPLE_DIR=%DIR%\..\public\samples
if not exist "%SAMPLE_DIR%" mkdir "%SAMPLE_DIR%"

REM Find first image in the samples directory
SET SAMPLE_IMAGE=
for /f "tokens=*" %%a in ('dir /b "%SAMPLE_DIR%\*.jpg" "%SAMPLE_DIR%\*.png" 2^>nul') do (
    if not defined SAMPLE_IMAGE set SAMPLE_IMAGE=%SAMPLE_DIR%\%%a
)

if not defined SAMPLE_IMAGE (
    echo No sample images found. Please place a test X-ray image in the public\samples directory.
) else (
    echo Using sample image: %SAMPLE_IMAGE%
    echo.
    echo Testing with predict_exact.py...
    python "%DIR%\predict_exact.py" --image "%SAMPLE_IMAGE%" --model "%FINAL_MODEL%" --fallback
)

echo.
echo Diagnostics complete.
pause 