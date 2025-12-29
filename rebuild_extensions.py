#!/usr/bin/env python3
"""
Rebuild TRELLIS extensions with correct GPU architecture.
"""
import os
import sys
import subprocess

# Set GPU architecture for AMD Radeon 8060S (gfx1151 / RDNA 3.5)
os.environ['GPU_ARCH'] = 'gfx1151'
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx1151'

print(f"GPU_ARCH={os.environ['GPU_ARCH']}")
print(f"PYTORCH_ROCM_ARCH={os.environ['PYTORCH_ROCM_ARCH']}")

extensions = [
    'extensions/diff-gaussian-rasterization',
    'extensions/nvdiffrast-hip',
    'extensions/torchsparse',
]

for ext_path in extensions:
    full_path = os.path.join(os.path.dirname(__file__), ext_path)
    print(f"\n{'='*60}")
    print(f"Building: {ext_path}")
    print('='*60)

    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-e', '.', '--no-build-isolation', '--force-reinstall', '-v'],
        cwd=full_path,
        env=os.environ
    )

    if result.returncode != 0:
        print(f"FAILED: {ext_path}")
        sys.exit(1)
    print(f"SUCCESS: {ext_path}")

print("\n" + "="*60)
print("All extensions rebuilt successfully!")
print("="*60)
