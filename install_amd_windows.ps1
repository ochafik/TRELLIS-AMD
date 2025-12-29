# TRELLIS-AMD Installation Script for Windows (PowerShell)
# For AMD Radeon RX 7000/9000 and Ryzen AI APUs with ROCm
#
# Usage: .\install_amd_windows.ps1 [-GpuArch gfx1151] [-UseUv] [-Force]
#
# Based on:
#   https://github.com/ROCm/TheRock/blob/main/RELEASES.md
#   https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/

param(
    [string]$GpuArch = "gfx1151",
    [switch]$UseUv,
    [switch]$Force
)

# Helper function for idempotent pip installs
function Install-PyPackage {
    param([string]$Package, [string[]]$ExtraArgs = @())

    # Check if already installed (for simple package names)
    if (-not $Force -and $Package -notmatch "^https?://") {
        $pkgName = $Package -replace "[<>=!].*", ""
        $installed = python -c "import $pkgName" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  $pkgName already installed, skipping" -ForegroundColor Gray
            return
        }
    }

    if ($script:pipCmd -eq "uv") {
        & uv pip install @ExtraArgs $Package
    } else {
        & pip install @ExtraArgs $Package
    }
}

$ErrorActionPreference = "Stop"

Write-Host "=============================================="
Write-Host "  TRELLIS-AMD Installation Script for Windows"
Write-Host "  For AMD GPUs with ROCm (via TheRock)"
Write-Host "=============================================="
Write-Host ""

# Check for Python 3.12 (required by AMD ROCm wheels)
$pythonCmd = $null
$pythonArgs = @()

# Try py -3.12 first (Windows Launcher with version)
try {
    $ver = py -3.12 --version 2>&1
    if ($ver -match "Python 3\.12") {
        $pythonCmd = "py"
        $pythonArgs = @("-3.12")
    }
} catch {}

# Fall back to python3.12
if (-not $pythonCmd) {
    try {
        $ver = python3.12 --version 2>&1
        if ($ver -match "Python 3\.12") {
            $pythonCmd = "python3.12"
        }
    } catch {}
}

# Fall back to python and check version
if (-not $pythonCmd) {
    try {
        $ver = python --version 2>&1
        if ($ver -match "Python 3\.12") {
            $pythonCmd = "python"
        }
    } catch {}
}

if (-not $pythonCmd) {
    Write-Host "ERROR: Python 3.12 not found. AMD ROCm wheels require Python 3.12." -ForegroundColor Red
    Write-Host "Install from: https://python.org/downloads/"
    Write-Host ""
    Write-Host "Available Python versions:"
    py --list 2>&1 | Write-Host
    exit 1
}

$pythonVersion = & $pythonCmd @pythonArgs --version 2>&1
Write-Host "Detected $pythonVersion (using '$pythonCmd $pythonArgs')"

# Set GPU architecture
$env:GPU_ARCH = $GpuArch
$env:PYTORCH_ROCM_ARCH = $GpuArch
$env:HIP_VISIBLE_DEVICES = "0"

# IMPORTANT: Unset this - Python 3.12 removed distutils from stdlib
Remove-Item Env:SETUPTOOLS_USE_DISTUTILS -ErrorAction SilentlyContinue

Write-Host "Using GPU architecture: $GpuArch"
Write-Host ""
Write-Host "Supported architectures:"
Write-Host "  - RX 7900 XTX/XT: gfx1100"
Write-Host "  - RX 7800 XT/7700 XT: gfx1101"
Write-Host "  - RX 9070 XT: gfx1201"
Write-Host "  - Strix Halo APU: gfx1151 (default)"
Write-Host ""

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Check for spaces in path
if ($ScriptDir -match ' ') {
    Write-Host "WARNING: Your path contains spaces: $ScriptDir" -ForegroundColor Yellow
    Write-Host "This may cause issues with some build tools, but we'll try anyway."
    Write-Host "If builds fail, consider cloning to C:\dev\TRELLIS-AMD"
    Write-Host ""
}

# Step 1: Create virtual environment
Write-Host "[1/7] Creating Python virtual environment..."
if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "Creating new virtual environment with Python 3.12..."
    # Try normal venv first, fall back to --without-pip if ensurepip fails
    & $pythonCmd @pythonArgs -m venv .venv 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ensurepip failed, creating venv without pip..."
        & $pythonCmd @pythonArgs -m venv .venv --without-pip
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: Failed to create virtual environment"
            exit 1
        }
    }
}
& ".\.venv\Scripts\Activate.ps1"

# Detect or install uv for faster installs
$script:pipCmd = "pip"
if ($UseUv -or (Get-Command uv -ErrorAction SilentlyContinue)) {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Host "Installing uv..."
        pip install uv
    }
    $script:pipCmd = "uv"
    Write-Host "Using uv for faster installs" -ForegroundColor Green
}

# Ensure pip is installed (in case we used --without-pip)
if (-not (Get-Command pip -ErrorAction SilentlyContinue)) {
    Write-Host "Installing pip via get-pip.py..."
    Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile "get-pip.py"
    python get-pip.py --no-warn-script-location
    Remove-Item "get-pip.py"
}

# Step 2: Upgrade pip and install build tools
Write-Host ""
Write-Host "[2/7] Upgrading pip and installing build tools..."
if ($script:pipCmd -eq "uv") {
    uv pip install --upgrade pip setuptools wheel ninja
} else {
    python -m pip install --upgrade pip
    python -m pip install --upgrade setuptools wheel ninja
}


# Step 3: Install ROCm SDK + PyTorch from TheRock nightlies
Write-Host ""
Write-Host "[3/7] Installing ROCm SDK and PyTorch..."

$rocmIndex = "https://rocm.nightlies.amd.com/v2/$GpuArch/"
Write-Host "  Using index: $rocmIndex"

# Check if already installed
$oldErrorPref = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$null = python -c "import torch; import rocm_sdk" 2>&1
$alreadyInstalled = $LASTEXITCODE -eq 0
$ErrorActionPreference = $oldErrorPref

if (-not $Force -and $alreadyInstalled) {
    Write-Host "  ROCm SDK and PyTorch already installed, skipping" -ForegroundColor Gray
} else {
    Write-Host "  Installing ROCm SDK (~3.5GB, this may take a while)..."
    if ($script:pipCmd -eq "uv") {
        uv pip install --index-url $rocmIndex "rocm[libraries,devel]"
    } else {
        pip install --index-url $rocmIndex "rocm[libraries,devel]"
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: ROCm SDK installation had issues, continuing..." -ForegroundColor Yellow
    }

    Write-Host "  Installing PyTorch..."
    if ($script:pipCmd -eq "uv") {
        uv pip install --index-url $rocmIndex --pre torch torchvision torchaudio
    } else {
        pip install --index-url $rocmIndex --pre torch torchvision torchaudio
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: PyTorch installation failed" -ForegroundColor Red
        exit 1
    }
}

# Verify PyTorch
Write-Host ""
Write-Host "Verifying PyTorch installation..."
python -c "import torch; print('PyTorch', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Step 4: Install TRELLIS dependencies
Write-Host ""
Write-Host "[4/7] Installing TRELLIS Python dependencies..."
if ($script:pipCmd -eq "uv") {
    uv pip install -r requirements.txt
} else {
    pip install -r requirements.txt
}
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Some requirements may have failed. Continuing..." -ForegroundColor Yellow
}

# Step 5: Install nvdiffrast-hip
Write-Host ""
Write-Host "[5/7] Installing nvdiffrast-hip..."
$oldErrorPref = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$null = python -c "import nvdiffrast" 2>&1
$nvdiffrastInstalled = $LASTEXITCODE -eq 0
$ErrorActionPreference = $oldErrorPref

if (-not $Force -and $nvdiffrastInstalled) {
    Write-Host "  nvdiffrast already installed, skipping" -ForegroundColor Gray
} else {
    $extPath = Join-Path $ScriptDir "extensions\nvdiffrast-hip"
    if ($script:pipCmd -eq "uv") {
        uv pip install $extPath --no-build-isolation -v
    } else {
        pip install $extPath --no-build-isolation -v
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: nvdiffrast-hip installation failed." -ForegroundColor Yellow
        Write-Host "You may need Visual Studio 2022 with C++ Build Tools installed."
    }
}

# Step 6: Build diff-gaussian-rasterization
Write-Host ""
Write-Host "[6/7] Building diff-gaussian-rasterization..."
$oldErrorPref = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$null = python -c "import diff_gaussian_rasterization" 2>&1
$diffGaussianInstalled = $LASTEXITCODE -eq 0
$ErrorActionPreference = $oldErrorPref

if (-not $Force -and $diffGaussianInstalled) {
    Write-Host "  diff_gaussian_rasterization already installed, skipping" -ForegroundColor Gray
} else {
    $extPath = Join-Path $ScriptDir "extensions\diff-gaussian-rasterization"
    Write-Host "Attempting build with GPU_ARCH=$GpuArch..."
    if ($script:pipCmd -eq "uv") {
        uv pip install $extPath --no-build-isolation -v
    } else {
        pip install $extPath --no-build-isolation -v
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: diff-gaussian-rasterization build may have failed." -ForegroundColor Yellow
    }
}

# Step 7: Install torchsparse
Write-Host ""
Write-Host "[7/7] Installing torchsparse..."
$oldErrorPref = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$null = python -c "import torchsparse" 2>&1
$torchsparseInstalled = $LASTEXITCODE -eq 0
$ErrorActionPreference = $oldErrorPref

if (-not $Force -and $torchsparseInstalled) {
    Write-Host "  torchsparse already installed, skipping" -ForegroundColor Gray
} else {
    $extPath = Join-Path $ScriptDir "extensions\torchsparse"
    $env:CUDA_HOME = ""
    $env:FORCE_CUDA = "1"
    if ($script:pipCmd -eq "uv") {
        uv pip install $extPath --no-build-isolation -v
    } else {
        pip install $extPath --no-build-isolation -v
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: torchsparse installation failed." -ForegroundColor Yellow
        Write-Host "The app may still work with CPU-only sparse operations."
    }
}

Write-Host ""
Write-Host "=============================================="
Write-Host "  Installation Complete!"
Write-Host "=============================================="
Write-Host ""
Write-Host "To run TRELLIS:"
Write-Host ""
Write-Host '  .\.venv\Scripts\Activate.ps1'
Write-Host '  $env:ATTN_BACKEND = "sdpa"'
Write-Host '  $env:XFORMERS_DISABLED = "1"'
Write-Host '  $env:SPARSE_BACKEND = "torchsparse"'
Write-Host '  python app.py'
Write-Host ""
Write-Host "Then open http://localhost:7860 in your browser"
Write-Host ""
Write-Host "NOTE: GLB export takes 5-10 minutes - this is normal!"
Write-Host "      Gaussian export is much faster (~30 seconds)."
Write-Host ""
