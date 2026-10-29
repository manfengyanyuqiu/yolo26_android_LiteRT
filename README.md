# 天玑 9500 NPU 推理与模型转换

这是一个面向联发科天玑 9500 的 Android 目标检测示例，使用 LiteRT 的 MediaTek Neuron NPU 加速路径。项目包含 Android 推理端和独立的 YOLO 模型转换工具。 
## 功能

- YOLOv5、YOLOv8、YOLO26 检测模型
- TFLite FP32、FP16 权重 + FLOAT32 I/O、INT8 输入输出
- 图片检测与手机后置摄像头实时检测
- Camera2 constrained high speed 摄像头路径，选择设备公开的固定 120 FPS 模式；设备不公开该模式时直接报错
- 预处理、NPU 执行、同步读回、后处理和每次推理耗时显示

项目已移除 OBS、UDP、MJPEG 图传接收代码。输入来自本地图片或手机摄像头，不包含网络视频传输功能。

## 平台边界

Android 推理端当前使用 **MediaTek Neuron/LiteRT NPU** 加速，模型加载时会验证 MediaTek NPU 和编译插件，并且不会静默切换 CPU。高通平台需要自行接入对应版本的 **QNN Delegate**，同时适配输入输出张量、量化参数、图编译和后处理验证；本项目不宣称支持高通平台。

## 构建 Android 应用

需要 Android SDK、JDK 17 和 Gradle 8.13。离线构建命令：

```powershell
$env:JAVA_HOME="C:\Path\to\jdk-17"
$env:GRADLE_USER_HOME="C:\Path\to\gradle-cache"
.\.toolchain\gradle-8.13\bin\gradle.bat --offline testDebugUnitTest assembleDebug lintDebug
```

APK 输出在 `app/build/outputs/apk/debug/app-debug.apk`。手机需要 Android 12（API 31）或更高、arm64，以及可用的 MediaTek Neuron/LiteRT 驱动。

## 模型转换工具

工具位于 [`model-exporter/`](model-exporter/)，支持 YOLOv5/v8/26 的 PT 或解码 ONNX 到 TFLite，输出 FP32、FP16 权重和 INT8。输出尺寸必须是 32 的倍数，默认 320×320；输出不能包含内置 NMS。INT8 导出需要至少 32 张代表性校准图片。

复制 `model-exporter/exporter-runtime.example.json` 为 `exporter-runtime.json`，填写本机 Python 环境路径，再运行 `model-exporter/launch_exporter.cmd`。依赖和命令示例见 [`model-exporter/README.md`](model-exporter/README.md)。

## 速度测试

`tools/benchmark_latency.py` 会交替运行原始/优化后处理并检查输出完全一致：

```powershell
python tools/benchmark_latency.py --model path/to/model.tflite `
  --family YOLO26 --layout CHANNELS --output .deps/latency-result
```

测速数据应区分预热和冷启动，并注明是否包含图片解码、摄像头采集和屏幕显示。

天玑 9500 上 v5/v8/YOLO26 的 n/s、FP16 和 INT8 实测数据见 SPEED_BENCHMARK.md。

## 许可证

YOLOv5、Ultralytics/YOLO26 相关代码和模型遵循其各自上游许可证，见 `model-exporter/LICENSE-*.txt`。发布模型时请保留源模型许可证和来源信息。
