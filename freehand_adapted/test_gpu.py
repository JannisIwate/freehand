#!/usr/bin/env python
"""
Quick GPU detection test for the FreeHand project
"""
import torch

print("=" * 60)
print("FreeHand GPU Configuration Test")
print("=" * 60)
print()

print("PyTorch Information:")
print(f"  Version: {torch.__version__}")
print(f"  CUDA Available: {torch.cuda.is_available()}")
print()

if torch.cuda.is_available():
    print("GPU Configuration:")
    print(f"  Device Count: {torch.cuda.device_count()}")
    print(f"  Current Device: {torch.cuda.current_device()}")
    print(f"  Device Name: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"  Total Memory: {props.total_memory / 1e9:.2f} GB")
    print(f"  Memory Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    print(f"  Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print()
    print("Status: GPU is ready for training!")
else:
    print("⚠ Warning: CUDA is NOT available")
    print()
    print("To enable GPU support, follow these steps:")
    print("1. Open Command Prompt/PowerShell")
    print("2. Run: INSTALL_PYTORCH_CUDA.bat")
    print("   OR")
    print("   conda install -c pytorch -c conda-forge pytorch::pytorch pytorch::pytorch-cuda=11.8 -y")
    print()
    print("For more details, see: GPU_SETUP.md")

print()
print("=" * 60)
