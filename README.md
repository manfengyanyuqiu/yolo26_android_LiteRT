# 🧠 YOLO Android NPU 推理与模型转换

这是一个面向联发科天玑 9500 的 Android 目标检测项目。Android 端使用 **Java 17** 编写，调用 LiteRT 的 MediaTek Neuron NPU 运行时完成 YOLOv5、YOLOv8 和 YOLO26 推理；仓库同时提供一个 Python 模型转换工具，把 PT/ONNX 模型转换成适合手机端部署的 TFLite FP32、FP16 或 INT8 文件。

Author:漫逢烟雨秋
QQ：2542588178

觉得项目不错的兄弟点点star

## 📋 当前状态

- 支持 YOLOv5、YOLOv8、YOLO26 的检测输出。
- 支持 224x224 和 320x320 等 32 的倍数输入尺寸。
- 支持 FP32、FP16 权重 + FLOAT32 I/O、INT8 TFLite 模型。
- 图片检测和后置摄像头实时检测均走同一套 Neuron 推理封装。
- Camera2 使用设备公开的 constrained high-speed 模式；当前测试设备提供 120 FPS 模式。
- 界面显示每次推理耗时，并在摄像头预览上绘制检测框。
- NPU 模型加载时检查 MediaTek Neuron 编译插件，不静默切换到 CPU。


### 📱 Android 推理端

- **Java 17**：应用、NPU 调用、摄像头管线和后处理均为 Java；没有 Kotlin 业务源码。
- Android SDK API 36，最低 API 31，目标 API 36。
- LiteRT 2.3.0 与 `litert-npu-runtime-mediatek:2.3.0`。
- MediaTek Neuron/LiteRT NPU runtime。
- Camera2、OpenGL ES 2.0、Gradle Kotlin DSL。
- 仅构建 `arm64-v8a`，适用于带 MediaTek Neuron 驱动的设备。

### 🐍 模型转换端

- **Python**：模型转换、ONNX/TFLite 检查、校准集处理和主机精度验证脚本。
- Ultralytics、ONNX Runtime、onnx2tf、TensorFlow Lite 等转换依赖由 `model-exporter/requirements-*.txt` 管理。

GitHub 的语言统计因此应同时包含 Java 和 Python：Java 是 Android 主体，Python 是独立的模型转换和验证工具。

## 📂 Java 模块

Android Java 源码位于 `app/src/main/java/`：

| 文件 | 职责 |
| --- | --- |
| `NpuDetector.java` | 检查模型元数据、编译 Neuron 模型、预处理输入、执行 NPU、读回输出 |
| `PreparedNpuCall.java` | 复用 LiteRT prepared call，减少重复分配和编译开销 |
| `NpuOnlyModel.java` | NPU-only 模型封装，明确禁止 CPU 回退 |
| `YoloDecoder.java` | YOLOv5/v8/26 输出读取、候选筛选、排序和 NMS |
| `RgbaInput.java` | 摄像头 RGBA 帧到模型输入张量的打包 |
| `Camera2Source.java` | Camera2 high-speed 会话和 120 FPS 帧采集 |
| `CameraGlView.java` | OpenGL ES 预览渲染 |
| `DetectionOverlay.java` | 将检测框映射到预览坐标并绘制 |
| `MainActivity.java` | 图片选择、模型选择和检测入口 |
| `CameraDetectionActivity.java` | 摄像头检测工作流和推理耗时显示 |

单元测试和设备测试分别位于 `app/src/test/java/` 与 `app/src/androidTest/java/`，覆盖解码器、后处理一致性、模型调用和真机 NPU 延迟。

## 📁 目录结构

```text
app/src/main/java/       Android Java 推理、Camera2 和画框代码
app/src/test/java/       主机单元测试
app/src/androidTest/java/ 设备测试与真机测速
model-exporter/          独立 Python 模型转换工具
tools/                   转换、校验和延迟测试脚本
SPEED_BENCHMARK.md       完整测速原始记录
MODEL_EXPORTER.md        模型转换流程说明
CAMERA.md                Camera2 采集说明
```

## 🔨 构建 Android 应用

环境要求：Android SDK、JDK 17、Gradle 8.13。项目默认使用本地 Gradle wrapper 和离线缓存；没有本地依赖缓存时，先按 `app/build.gradle.kts` 配置的版本准备依赖。

```powershell
$env:JAVA_HOME="C:\Path\to\jdk-17"
.\gradlew.bat testDebugUnitTest assembleDebug lintDebug
```

APK 输出路径：`app/build/outputs/apk/debug/app-debug.apk`。

设备需要 Android 12（API 31）或更高版本、`arm64-v8a` ABI，以及可用的 MediaTek Neuron/LiteRT 驱动。安装后从图片检测入口选择 TFLite 模型，或进入摄像头检测页面查看实时画框和单次推理耗时。

## 🔄 模型转换

模型转换工具位于 [`model-exporter/`](model-exporter/)。它支持 YOLOv5、YOLOv8、YOLO26 的 PT 或 ONNX 输入，并输出 FP32、FP16 权重和 INT8 全整数 TFLite。输出尺寸必须是 32 的倍数，默认使用 320x320；输出模型不应包含内置 NMS。

先复制运行时配置并填写本机 Python 环境：

```powershell
Copy-Item model-exporter/exporter-runtime.example.json model-exporter/exporter-runtime.json
.\model-exporter\launch_exporter.cmd
```

INT8 导出至少需要 32 张具有代表性的 JPG/PNG 校准图片。转换流程会生成阶段日志、张量报告、主机精度对照和 SHA256 清单。主机转换通过后，仍需在手机上单独执行 Neuron 编译和真机验证。

更多参数和 ONNX 布局说明见 [`model-exporter/README.md`](model-exporter/README.md) 与 [`MODEL_EXPORTER.md`](MODEL_EXPORTER.md)。

## ⚡ 天玑 9500 真机测速

测试日期：2026-10-30。设备：vivo V2548A / MT6993 / Android 17。后端：LiteRT 2.3.0 MediaTek Neuron，NPU-only。全部模型统一使用 **320×320** 输入，同一张 640×480 图片，每项指标分别预热 5 次、采样 50 次，表中为算术平均耗时。

| 模型 | 精度 | 输入尺寸（px） | NPU 执行及读回平均耗时（ms） | 图片处理平均耗时（ms） |
| --- | --- | --- | ---: | ---: |
| YOLOv5n | FP16 权重 / FLOAT32 I/O | 320×320 | 4.94 | 6.00 |
| YOLOv5n | INT8 | 320×320 | 1.97 | 3.57 |
| YOLOv5s | FP16 权重 / FLOAT32 I/O | 320×320 | 6.19 | 7.33 |
| YOLOv5s | INT8 | 320×320 | 3.09 | 4.72 |
| YOLOv8n | FP16 权重 / FLOAT32 I/O | 320×320 | 3.07 | 5.64 |
| YOLOv8n | INT8 | 320×320 | 1.59 | 5.30 |
| YOLOv8s | FP16 权重 / FLOAT32 I/O | 320×320 | 6.15 | 8.18 |
| YOLOv8s | INT8 | 320×320 | 2.34 | 6.53 |
| YOLO26n | FP16 权重 / FLOAT32 I/O | 320×320 | 4.19 | 6.69 |
| YOLO26n | INT8 | 320×320 | 1.34 | 5.39 |
| YOLO26s | FP16 权重 / FLOAT32 I/O | 320×320 | 5.93 | 8.38 |
| YOLO26s | INT8 | 320×320 | 2.50 | 7.40 |

NPU 执行及读回：从提交已准备的 NPU 调用开始，到同步读取输出完成。图片处理：从已解码的 Bitmap 开始，包含 letterbox、RGB/量化输入打包、NPU 执行、同步读回和 YOLO 解码/NMS。两项均不包含模型编译、JPEG 解码、摄像头采集或屏幕显示。

12 个模型均完成 NPU 编译和执行。FP16 指权重存储为 FP16、输入输出为 FLOAT32，硬件算术精度由 Neuron 驱动决定。速度表不代表精度评估：YOLOv8n INT8 存在明显数值偏差，YOLO26n/s INT8 在测试图上也产生大量候选框，部署前需用实际数据集验证检测精度。各模型依次测试，保留设备正常调频；温度、后台负载、驱动和图片内容均会影响耗时。

[机器可读结果、模型/图片 SHA256 与 P50/P95](benchmarks/2026-10-30-320.json)。高通设备需要自行接入 QNN Delegate，以上数据仅适用于本次 MediaTek 设备测试。

```powershell
python tools/benchmark_latency.py --model path/to/model.tflite `
  --family YOLO26 --layout CHANNELS --output .deps/latency-result --device-only
```

## ⚠️ 平台边界

当前 Android 推理路径绑定 MediaTek Neuron/LiteRT NPU runtime，模型加载会验证 MediaTek 编译插件，并且不会静默切换 CPU。高通平台需要自行接入对应版本的 **QNN Delegate**，适配输入输出张量、量化参数、图编译和 YOLO 后处理验证；本项目的 Neuron 测速数据不适用于高通设备。

## 📄 许可证

YOLOv5、Ultralytics/YOLO26 相关代码和模型遵循各自上游许可证，见 `model-exporter/LICENSE-*.txt`。发布转换后的模型时，请保留源模型许可证、来源和训练数据说明。
