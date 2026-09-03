# Sovereign AI Workbench - Automated Windows Setup and Launch Script
# PowerShell 5.1+ / 7+
[CmdletBinding()]
param(
    [switch]$SkipModelDownload,
    [string]$ModelChoice = "recommended",
    [switch]$NoStart
)

$ErrorActionPreference = "Stop"

Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "   Sovereign AI Workbench - Windows Setup" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check & Install Git, Python, Node.js if missing via winget
function Assert-OrInstall($command, $wingetId, $name) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        Write-Host "[-] $name is not detected in PATH." -ForegroundColor Yellow
        if (Get-Command winget -ErrorAction SilentlyContinue) {
            Write-Host "[*] Installing $name via Windows Package Manager (winget)..." -ForegroundColor Cyan
            winget install --id $wingetId -e --source winget --accept-source-agreements --accept-package-agreements
            # Refresh PATH for current session
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        } else {
            Write-Error "[!] $name is required. Please install it manually and re-run this script."
            exit 1
        }
    } else {
        Write-Host "[+] $name is installed." -ForegroundColor Green
    }
}

Assert-OrInstall "git" "Git.Git" "Git"
Assert-OrInstall "python" "Python.Python.3.11" "Python 3.11"
Assert-OrInstall "node" "OpenJS.NodeJS.LTS" "Node.js (LTS)"

# 2. Virtual Environment Setup
Write-Host ""
Write-Host "[1/5] Setting up Python Virtual Environment..." -ForegroundColor Cyan
if (-not (Test-Path "venv")) {
    python -m venv venv
    Write-Host "[+] Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "[+] Virtual environment already exists." -ForegroundColor Green
}

$venvPython = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
$venvPip = Join-Path $PSScriptRoot "venv\Scripts\pip.exe"
if (-not (Test-Path $venvPython)) {
    $venvPython = "venv\Scripts\python.exe"
    $venvPip = "venv\Scripts\pip.exe"
}

Write-Host "[*] Upgrading pip, setuptools, wheel..." -ForegroundColor Gray
& $venvPython -m pip install --upgrade pip setuptools wheel --quiet

# 3. Detect GPU & Install llama-cpp-python prebuilt wheels + requirements
Write-Host ""
Write-Host "[2/5] Installing Backend Dependencies..." -ForegroundColor Cyan

$hasGpu = $false
try {
    $smi = Start-Process -FilePath "nvidia-smi" -ArgumentList "--query-gpu=name", "--format=csv,noheader" -NoNewWindow -Wait -PassThru -ErrorAction SilentlyContinue
    if ($smi.ExitCode -eq 0) {
        $hasGpu = $true
    }
} catch {
    $hasGpu = $false
}

if ($hasGpu) {
    Write-Host "[+] NVIDIA GPU detected. Installing llama-cpp-python with CUDA acceleration..." -ForegroundColor Green
    & $venvPip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --no-warn-script-location
} else {
    Write-Host "[-] No NVIDIA GPU detected. Installing llama-cpp-python with optimized CPU support..." -ForegroundColor Yellow
    & $venvPip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --no-warn-script-location
}

& $venvPip install -r requirements.txt huggingface_hub --no-warn-script-location
Write-Host "[+] Python backend dependencies installed." -ForegroundColor Green

# 4. Frontend Dependencies
Write-Host ""
Write-Host "[3/5] Installing Frontend Dependencies..." -ForegroundColor Cyan
Push-Location "frontend"
npm install
Pop-Location
Write-Host "[+] Frontend dependencies installed." -ForegroundColor Green

# 5. Environment File
Write-Host ""
Write-Host "[4/5] Configuring Environment..." -ForegroundColor Cyan
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "[+] Created .env from .env.example" -ForegroundColor Green
    }
} else {
    Write-Host "[+] .env file already present." -ForegroundColor Green
}

# 6. Model Download
Write-Host ""
Write-Host "[5/5] Checking Model Weights..." -ForegroundColor Cyan
if (-not $SkipModelDownload) {
    & $venvPython scripts\download_models.py --model $ModelChoice
} else {
    Write-Host "[*] Skipping model download per flag." -ForegroundColor Gray
}

Write-Host ""
Write-Host "===================================================" -ForegroundColor Green
Write-Host "   Setup Completed Successfully!" -ForegroundColor Green
Write-Host "===================================================" -ForegroundColor Green
Write-Host ""

if (-not $NoStart) {
    Write-Host "[*] Launching Backend API and Frontend UI in separate windows..." -ForegroundColor Cyan
    
    # Launch Backend
    Start-Process cmd.exe -ArgumentList '/k "title Sovereign AI Backend && call venv\Scripts\activate.bat && python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"'
    
    # Launch Frontend
    Start-Process cmd.exe -ArgumentList '/k "title Sovereign AI Frontend && cd frontend && npm run dev -- --host 0.0.0.0"'
    
    Start-Sleep -Seconds 3
    Write-Host "[*] Opening browser at http://localhost:5173 ..." -ForegroundColor Green
    Start-Process "http://localhost:5173"
} else {
    Write-Host "Run .\start_windows.bat to launch the workbench at any time." -ForegroundColor Cyan
}
