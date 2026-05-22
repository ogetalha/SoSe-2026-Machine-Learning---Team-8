param(
    [string]$BagDir = "",
    [string]$ExtractedImagesDir = "extracted_images",
    [string]$VisualizationOutputDir = "visualization_output",
    [int]$Max = 0,
    [switch]$NoPreview,
    [switch]$StartCompose
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Get-FirstBagDir {
    $bagsRoot = Join-Path $PSScriptRoot "bags"
    if (-not (Test-Path $bagsRoot)) {
        throw "Bags folder not found: $bagsRoot"
    }

    $candidate = Get-ChildItem -Path $bagsRoot -Directory |
        Where-Object { Test-Path (Join-Path $_.FullName "metadata.yaml") } |
        Sort-Object Name |
        Select-Object -First 1

    if (-not $candidate) {
        throw "No valid bag folder found in ./bags (expected metadata.yaml in a subfolder)."
    }

    return $candidate.FullName
}

function Get-VenvPython {
    $venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return $venvPython
    }

    # Fallback to check parent folder for .venv if running inside a worktree
    $parentVenvPython = Join-Path $PSScriptRoot "..\..\.venv\Scripts\python.exe"
    if (Test-Path $parentVenvPython) {
        return $parentVenvPython
    }

    Write-Step "Creating Python virtual environment (.venv)"
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        & py -3 -m venv (Join-Path $PSScriptRoot ".venv")
    }
    else {
        $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
        if (-not $pythonCmd) {
            throw "Python was not found. Install Python 3 and try again."
        }
        & python -m venv (Join-Path $PSScriptRoot ".venv")
    }

    if (-not (Test-Path $venvPython)) {
        throw "Failed to create .venv"
    }

    return $venvPython
}

function Ensure-Package {
    param(
        [string]$PythonExe,
        [string]$ImportName,
        [string]$PipName
    )

    $checkCode = "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$ImportName') else 1)"
    & $PythonExe -c $checkCode
    if ($LASTEXITCODE -ne 0) {
        Write-Step "Installing missing package: $PipName"
        & $PythonExe -m pip install $PipName
    }
    else {
        Write-Host "Package already installed: $PipName" -ForegroundColor Green
    }
}

Set-Location $PSScriptRoot
Write-Step "Working directory: $PSScriptRoot"

if ($StartCompose) {
    Write-Step "Starting Docker Compose (bag profile)"
    & docker compose --profile bag up --build -d
}

$pythonExe = Get-VenvPython
Write-Step "Upgrading pip"
& $pythonExe -m pip install --upgrade pip

Write-Step "Checking required Python packages"
Ensure-Package -PythonExe $pythonExe -ImportName "cv2" -PipName "opencv-python"
Ensure-Package -PythonExe $pythonExe -ImportName "rosbags" -PipName "rosbags"
Ensure-Package -PythonExe $pythonExe -ImportName "numpy" -PipName "numpy"
Ensure-Package -PythonExe $pythonExe -ImportName "sklearn" -PipName "scikit-learn"

if ([string]::IsNullOrWhiteSpace($BagDir)) {
    $BagDir = Get-FirstBagDir
}

Write-Step "Using bag directory: $BagDir"

Write-Step "Extracting images from bag"
$imageCount = 0
if (Test-Path $ExtractedImagesDir) {
    $imageCount = (Get-ChildItem -Path $ExtractedImagesDir -Filter *.jpg).Count
}
# Fallback to desktop path if images are not extracted locally
if ($imageCount -lt 100) {
    $desktopImagesDir = "C:\Users\G7\Desktop\SoSe-2026-Machine-Learning---Team-8\extracted_images"
    if (Test-Path $desktopImagesDir) {
        $desktopImageCount = (Get-ChildItem -Path $desktopImagesDir -Filter *.jpg).Count
        if ($desktopImageCount -ge 100) {
            $ExtractedImagesDir = $desktopImagesDir
            $imageCount = $desktopImageCount
        }
    }
}

if ($imageCount -ge 100) {
    Write-Host "Found $imageCount images already in $ExtractedImagesDir. Skipping extraction." -ForegroundColor Green
} else {
    & $pythonExe ".\docs\extract_images_from_bag.py" $BagDir --out $ExtractedImagesDir
    if ($LASTEXITCODE -ne 0) {
        throw "Image extraction failed."
    }
}

Write-Step "Running steering visualization"
$vizArgs = @(
    ".\visualize_line_follower.py",
    "--images", $ExtractedImagesDir,
    "--output", $VisualizationOutputDir
)
if ($Max -gt 0) {
    $vizArgs += @("--max", $Max)
}
if (-not $NoPreview) {
    $vizArgs += "--show"
}
& $pythonExe @vizArgs
if ($LASTEXITCODE -ne 0) {
    throw "Visualization failed."
}

$csvPath = Join-Path $VisualizationOutputDir "steering.csv"
if (-not (Test-Path $csvPath)) {
    throw "Visualization completed but steering.csv was not created at: $csvPath"
}

Write-Step "Done"
Write-Host "Annotated frames: $VisualizationOutputDir\frames"
Write-Host "Steering CSV: $VisualizationOutputDir\steering.csv"
