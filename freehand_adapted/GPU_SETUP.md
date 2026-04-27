# GPU Setup for Training

## Current Status

Your system has:
- ✓ NVIDIA GeForce RTX 3090 GPU (24GB VRAM)
- ✓ NVIDIA Driver: 595.02
- ✓ CUDA Version: 13.0

However:
- ✗ PyTorch is installed with CPU-only support (torch 1.11.0+cpu)

## How to Enable GPU Support

### Option 1: Automatic Installation (Recommended)

Run the provided batch script:
```
INSTALL_PYTORCH_CUDA.bat
```

This will install PyTorch with CUDA 11.8 support automatically.

### Option 2: Manual Installation with Conda

Open Command Prompt or PowerShell and run:

```bash
conda activate FUS
conda install -c pytorch -c conda-forge pytorch::pytorch pytorch::pytorch-cuda=11.8 torchvision::torchvision -y
```

### Option 3: Manual Installation with Pip

```bash
# Activate the environment
conda activate FUS

# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Option 4: Download from Official Website

Visit https://pytorch.org/get-started/locally/ and select:
- PyTorch Build: Stable
- Operating System: Windows
- Package: pip or conda
- Language: Python
- Compute Platform: CUDA 11.8

## Verification

After installation, verify CUDA is working:

```python
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

You should see:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3090
```

## Training with GPU

Once PyTorch is installed with CUDA support, the training script (`scripts/train.py`) will automatically:
- Detect the GPU
- Display GPU information (name, memory, etc.)
- Use the GPU for training (much faster than CPU!)

The script will print something like:
```
GPU detected: NVIDIA GeForce RTX 3090
CUDA device count: 1
GPU Memory: 24.00 GB
Using device: cuda
```

## Performance Impact

Training with RTX 3090 GPU will be significantly faster than CPU:
- **Expected speedup: 50-100x faster** compared to CPU training
- On RTX 3090: ~100-200ms per batch
- On CPU: ~5-10 seconds per batch

## Troubleshooting

If you still see "CUDA not available" after installation:

1. Check NVIDIA driver is properly installed:
   ```
   nvidia-smi
   ```

2. Verify PyTorch CUDA version matches your CUDA driver:
   - Driver CUDA version: 13.0
   - Install PyTorch with CUDA 11.8 or 12.1

3. Restart Python/Jupyter after installation

4. Clear PyTorch cache:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

## Notes

- The current PyTorch version is 1.11.0 (from 2022)
- Consider updating to a newer version (2.0+) for better performance and features
- Ensure GPU drivers are kept updated from https://www.nvidia.com/Download/driverDetails.aspx
