# TRELLIS-AMD

**TRELLIS running on AMD GPUs with ROCm** - Image to 3D Asset Generation

This is a fork of [Microsoft TRELLIS](https://github.com/microsoft/TRELLIS) modified to run on AMD consumer GPUs (tested on RX 7800 XT with ROCm 6.4.2).

## Features

| Feature | Status | Timing |
|---------|--------|--------|
| ✅ 3D Model Generation | Working | ~45 seconds |
| ✅ Gaussian Splatting | Working (145+ it/s) | ~30 seconds |
| ✅ Gaussian Export (.ply) | Working | Instant |
| ✅ Mesh Extraction | Working | ~60 seconds |
| ✅ GLB Export with Textures | Working | **5-10 minutes** |

> **⚠️ GLB Export Takes 5-10 Minutes**: This is normal! The console will show progress through 5 steps. Your system will be under heavy load during texture baking - this is expected.

## Requirements

### Linux
- AMD GPU (tested: RX 7800 XT, RDNA3)
- ROCm 6.4+
- Python 3.10+
- ~16GB VRAM recommended
- System packages: `python3-venv`, `libsparsehash-dev`

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt install python3-venv python3-full libsparsehash-dev
```

### Windows (Experimental)
- AMD GPU: RX 7000/9000 series or Ryzen AI APUs (Strix Halo)
- Python 3.12 (required by AMD ROCm wheels)
- Visual Studio 2022 with C++ Build Tools
- ~16GB VRAM recommended

```powershell
# Install Python 3.12
winget install -e --id Python.Python.3.12

# Install Visual Studio Build Tools (if needed)
winget install -e --id Microsoft.VisualStudio.2022.BuildTools
```

## Quick Start

### Linux

```bash
# Clone the repository
git clone https://github.com/CalebisGross/TRELLIS-AMD
cd TRELLIS-AMD

# Run the installation script
chmod +x install_amd.sh
./install_amd.sh

# Activate environment and run
source .venv/bin/activate
ATTN_BACKEND=sdpa XFORMERS_DISABLED=1 SPARSE_BACKEND=torchsparse python app.py
```

### Windows

> **Note**: Windows support is experimental, based on AMD's new ROCm for Windows via [TheRock](https://github.com/ROCm/TheRock).

```powershell
# Clone to a path WITHOUT spaces (important!)
git clone https://github.com/CalebisGross/TRELLIS-AMD C:\dev\TRELLIS-AMD
cd C:\dev\TRELLIS-AMD

# For Strix Halo APU (gfx1151) - default
.\install_amd_windows.ps1

# For other GPUs, specify the architecture:
.\install_amd_windows.ps1 -GpuArch gfx1100  # RX 7900 XTX/XT
.\install_amd_windows.ps1 -GpuArch gfx1101  # RX 7800 XT
.\install_amd_windows.ps1 -GpuArch gfx1201  # RX 9070 XT

# Activate environment and run
.\.venv\Scripts\Activate.ps1
$env:ATTN_BACKEND = "sdpa"
$env:XFORMERS_DISABLED = "1"
$env:SPARSE_BACKEND = "torchsparse"
python app.py
```

Or use the batch file:
```cmd
set GPU_ARCH=gfx1151
install_amd_windows.bat
```

Then open http://localhost:7860 in your browser.

## What's Different from Original TRELLIS?

### Custom Extensions (AMD-compatible)

| Extension | Modification |
|-----------|-------------|
| **nvdiffrast-hip** | AMD-safe coarse rasterizer, HIP warp intrinsic macros |
| **diff-gaussian-rasterization** | Manual HIP build script, buffer initialization fixes |
| **torchsparse** | Built with `FORCE_CUDA=1` for HIP GPU backend |

### Application Modifications
- Switched to OpenGL rasterization backend (avoids HIP rasterizer bugs)
- Disabled `fill_holes` in mesh postprocessing (avoids visibility check issues)
- Added progress logging for GLB export

## Processing Time Reference

| Operation | Expected Time | Notes |
|-----------|--------------|-------|
| 3D Generation (Sampling) | ~45s | 12 steps of diffusion |
| Gaussian Export | Instant | Saves .ply file |
| GLB Export | **5-10 min** | Heavy CPU+GPU load is normal |

The GLB export shows progress in console:
```
[GLB Export] Starting GLB extraction (this takes 5-10 minutes)...
[GLB Export] Step 1/5: Mesh postprocessing...
[GLB Export] Step 2/5: UV parametrization...
[GLB Export] Step 3/5: Rendering multiview observations (100 views)...
[GLB Export] Step 4/5: Baking texture (2500 optimization steps)...
[GLB Export] Step 5/5: Finalizing GLB mesh...
[GLB Export] Complete!
```

## Known Limitations

1. **Mesh Preview**: May show grey - the actual export works correctly
2. **fill_holes Disabled**: Small holes in meshes may not be filled
3. **Performance**: Simplified coarse rasterizer is slower than NVIDIA-optimized version

## Troubleshooting

### Linux

#### GPU Hang/Crash
Ensure you're using ROCm 6.4+ and PyTorch built for ROCm.

#### Empty Mesh
Check that `fill_holes=False` is set in `trellis/utils/postprocessing_utils.py`.

#### CUDA Symbol Errors
Make sure you're using the AMD-modified extensions in this repo, not the original CUDA ones.

#### torchsparse "no attribute" Error
Rebuild with: `cd extensions/torchsparse && CUDA_HOME=/opt/rocm FORCE_CUDA=1 pip install . --no-build-isolation`

### Windows

#### Extension Build Fails
1. Install Visual Studio 2022 Build Tools with C++ workload
2. Make sure you're using Python 3.12 (required for ROCm Windows wheels)
3. Avoid paths with spaces - clone to `C:\dev\TRELLIS-AMD` instead

#### PyTorch Not Detecting GPU
1. Verify your GPU is supported (RX 7000/9000 series or Ryzen AI)
2. Install the latest AMD drivers
3. Check with: `python -c "import torch; print(torch.cuda.is_available())"`

#### "rocminfo" Not Found
The `rocm[devel]` package may not be available for all GPU architectures on Windows.
The extensions should still build using PyTorch's built-in HIP support.

#### Out of Memory Errors
Windows ROCm has known memory management issues on Ryzen AI APUs. Try:
- Close other GPU-intensive applications
- Reduce batch sizes in the application

#### Known Windows Limitations
- Only inference is fully supported (training may have issues)
- Only Python 3.12 is supported
- Some intermittent crashes may occur - this is expected with the preview ROCm stack

## Credits

- Original [TRELLIS](https://github.com/microsoft/TRELLIS) by Microsoft
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast) by NVIDIA
- AMD GPU modifications developed through extensive debugging of HIP compatibility issues

## License

See original licenses for TRELLIS, nvdiffrast, and diff-gaussian-rasterization.
