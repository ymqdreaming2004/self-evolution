$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot "train_venv\venv\Scripts\python.exe"
$SourcePath = Join-Path $ProjectRoot "src"
$BaseModel = Join-Path $ProjectRoot "train_venv\models\Qwen3.5-2B"
$Adapter = Join-Path $ProjectRoot "outputs\qwen35-fridge-qlora"

foreach ($RequiredPath in @($Python, $SourcePath, $BaseModel, $Adapter)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Required vision runtime path does not exist: $RequiredPath"
    }
}

$env:PYTHONPATH = $SourcePath
Set-Location -LiteralPath $ProjectRoot
& $Python -m uvicorn self_evolution_agent.vision_service:app --host 0.0.0.0 --port 8001
