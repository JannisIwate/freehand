@echo off
REM Script to install PyTorch with CUDA support
REM This fixes the "CUDA not available" issue

echo Installing PyTorch with CUDA 11.8 support...
echo.

REM Activate the FUS environment
call C:\Users\PrimaLab\anaconda3\Scripts\activate.bat C:\Users\PrimaLab\anaconda3\envs\FUS

REM Install PyTorch with CUDA support
REM Using conda forge as alternative channel
conda install -c pytorch -c conda-forge pytorch::pytorch pytorch::pytorch-cuda=11.8 torchvision::torchvision -y

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Successfully installed PyTorch with CUDA support!
    echo.
    echo Verifying installation...
    python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
) else (
    echo.
    echo Installation failed. Please try:
    echo 1. Download and run manually from: https://pytorch.org/get-started/locally/
    echo 2. Or use: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    echo.
)

pause
