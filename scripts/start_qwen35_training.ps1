$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot "train_venv\venv\Scripts\python.exe"
$LlamaFactory = Join-Path $ProjectRoot "train_venv\venv\Scripts\llamafactory-cli.exe"
$Config = Join-Path $ProjectRoot "outputs\qwen35-fridge-lora\training_args.yaml"
$OutputDir = Join-Path $ProjectRoot "outputs\qwen35-fridge-qlora"
$StdoutLog = Join-Path $OutputDir "train.stdout.log"
$StderrLog = Join-Path $OutputDir "train.stderr.log"
$PidFile = Join-Path $OutputDir "train.pid"
$env:HF_DATASETS_CACHE = Join-Path $ProjectRoot "train_venv\hf_cache_qwen35"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Training Python was not found: $Python"
}
if (-not (Test-Path -LiteralPath $LlamaFactory)) {
    throw "LLaMA-Factory CLI was not found: $LlamaFactory"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
& $Python (Join-Path $PSScriptRoot "launch_qwen35_training.py")
Write-Output "stdout: $StdoutLog"
Write-Output "stderr: $StderrLog"
