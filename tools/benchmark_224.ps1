param([string]$Serial)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$adbPath = Join-Path $projectRoot '.toolchain/sdk/platform-tools/adb.exe'
$deviceArgs = @()
if ($Serial) { $deviceArgs = @('-s', $Serial) }
function Invoke-Adb {
    & $adbPath @deviceArgs @args
    if ($LASTEXITCODE -ne 0) { throw "ADB failed: $args" }
}
Invoke-Adb get-state
Invoke-Adb install -r (Join-Path $projectRoot 'app/build/outputs/apk/debug/app-debug.apk')
Invoke-Adb install -r -t (Join-Path $projectRoot 'app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk')
Invoke-Adb push (Join-Path $projectRoot '../fydnn_demo/x64/Release/1.png') /data/local/tmp/fydnn-test.png
Invoke-Adb shell run-as com.fydnn.dimensity mkdir -p files
Invoke-Adb shell run-as com.fydnn.dimensity cp /data/local/tmp/fydnn-test.png files/test.png
$reportRoot = Join-Path $projectRoot '.deps/npu-debug/yolov5n-224'
New-Item -ItemType Directory -Force -Path $reportRoot | Out-Null
foreach ($name in @('yolov5n_224_4d_norm_fp32io_fp16weights', 'yolov5n_224_4d_norm_int8')) {
    $modelPath = Join-Path $projectRoot "models/yolov5n-224-mediatek/$name.tflite"
    Invoke-Adb push $modelPath /data/local/tmp/fydnn-model.tflite
    Invoke-Adb shell run-as com.fydnn.dimensity cp /data/local/tmp/fydnn-model.tflite "files/$name.tflite"
    Invoke-Adb push $modelPath "/sdcard/Download/$name.tflite"
    $output = Invoke-Adb shell am instrument -w -r -e model "$name.tflite" -e image test.png -e normalized true com.fydnn.dimensity.test/com.fydnn.dimensity.NpuDeviceInstrumentation
    $output | Set-Content -Encoding utf8 (Join-Path $reportRoot "$name-instrumentation.txt")
    $report = Invoke-Adb shell run-as com.fydnn.dimensity cat files/device-test.txt
    $report | Set-Content -Encoding utf8 (Join-Path $reportRoot "$name.txt")
    $report
    if (-not ($report -match '^PASS$')) { throw "NPU test failed for $name; see $reportRoot" }
}
Write-Output "Completed: $reportRoot"
