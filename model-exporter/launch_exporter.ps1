$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$settings = Join-Path $root 'exporter-runtime.json'
$repo = Split-Path -Parent $root
$python = Join-Path $repo '.toolchain/model-env/Scripts/python.exe'
if (Test-Path -LiteralPath $settings) {
    $config = Get-Content -LiteralPath $settings -Raw -Encoding UTF8 | ConvertFrom-Json
    $python = $config.convert_python
    if (-not [IO.Path]::IsPathRooted($python)) { $python = Join-Path $repo $python }
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'Python runtime is missing. Copy exporter-runtime.example.json to exporter-runtime.json and set convert_python.'
}
$pythonw = Join-Path (Split-Path -Parent $python) 'pythonw.exe'
if (-not (Test-Path -LiteralPath $pythonw -PathType Leaf)) {
    throw 'pythonw.exe is missing from the selected Python environment.'
}
$gui = Join-Path $root 'tools/model_exporter_gui.py'
Start-Process -FilePath $pythonw -ArgumentList ('"' + $gui + '"') -WorkingDirectory $root -WindowStyle Hidden
