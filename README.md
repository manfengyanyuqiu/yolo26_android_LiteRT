# 🧠 YOLO Android NPU 推理与模型转换

这是一个面向联发科天玑 9500 的 Android 目标检测项目。Android 端使用 **Java 17** 编写，调用 LiteRT 的 MediaTek Neuron NPU 运行时完成 YOLOv5、YOLOv8 和 YOLO26 推理；仓库同时提供一个 Python 模型转换工具，把 PT/ONNX 模型转换成适合手机端部署的 TFLite FP32、FP16 或 INT8 文件。

Author:漫逢烟雨秋

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

GitHub 的语言统计因此应同时包含 Java 和 Python：Java 是 Android 主体，Python 是独立的模型转换和验证工具。根目录 `.gitattributes` 已明确告诉 GitHub Linguist 如何识别两类源码。

## 📂 Java 模块

Android Java 源码位于 `app/src/main/java/com/fydnn/dimensity/`：

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

以下数据已合并到主 README，独立原始记录见 [`SPEED_BENCHMARK.md`](SPEED_BENCHMARK.md)。测试设备为 vivo V2548A（MT6993，Android 17），使用 MediaTek Neuron/LiteRT NPU，未启用 CPU 回退。同一张 640x480 JPEG，5 次预热、50 次采样；所有数字单位为毫秒，格式为 **平均 / P50 / P95**。

| 模型 | 精度 | 输入 | NPU/读回 | 图片端到端 |
| --- | --- | ---: | ---: | ---: |
| YOLOv5n | FP16 权重，FLOAT32 I/O | 224 | 3.44 / 3.50 / 4.24 | 4.37 / 4.30 / 5.17 |
| YOLOv5n | INT8 | 224 | 1.63 / 1.52 / 2.54 | 2.29 / 2.07 / 3.36 |
| YOLOv5s | FP16 权重，FLOAT32 I/O | 320 | 6.54 / 6.44 / 9.21 | 8.35 / 8.23 / 11.34 |
| YOLOv5s | INT8 | 320 | 2.40 / 2.45 / 3.12 | 3.96 / 3.76 / 6.17 |
| YOLOv8n | FP16 权重，FLOAT32 I/O | 320 | 3.03 / 3.01 / 4.21 | 4.72 / 4.63 / 5.27 |
| YOLOv8n | INT8 | 320 | 1.20 / 1.11 / 2.01 | 6.30 / 5.77 / 9.59 |
| YOLOv8s | FP16 权重，FLOAT32 I/O | 320 | 6.34 / 6.39 / 7.51 | 7.81 / 7.67 / 9.77 |
| YOLOv8s | INT8 | 320 | 2.28 / 2.27 / 2.61 | 5.13 / 5.12 / 5.85 |
| YOLO26n | FP16 权重，FLOAT32 I/O | 320 | 3.14 / 3.22 / 4.14 | 4.32 / 4.14 / 5.36 |
| YOLO26n | INT8 | 320 | 1.63 / 1.44 / 3.21 | 13.10 / 9.40 / 24.27 |
| YOLO26s | FP16 权重，FLOAT32 I/O | 320 | 6.01 / 5.78 / 8.19 | 6.18 / 6.26 / 7.53 |
| YOLO26s | INT8 | 320 | 2.11 / 1.92 / 3.30 | 12.85 / 12.65 / 15.34 |

NPU/读回包含一次 NPU 执行和同步输出读回；图片端到端还包含 JPEG 解码、letterbox/RGB 预处理和后处理，不包含摄像头采集、UDP 或屏幕显示。INT8 本轮编译和执行全部通过，但速度不代表精度与 FP16 一致，部署前应使用实际数据集检查精度。YOLOv8n INT8 的单图 NPU/主机数值对照曾出现明显偏差，需要单独校准和复核。

YOLOv8s FP16、YOLO26n/s 的旧基线后处理在当前测试图上候选数量较大，超过了单轮测试时限，因此表中报告实际优化链路和 NPU 数据，不把超时旧实现作为速度对照。FP16 在本项目中表示 FP16 权重存储、FLOAT32 输入输出，实际算术精度由 Neuron 驱动决定。

复测示例：

```powershell
python tools/benchmark_latency.py --model path/to/model.tflite `
  --family YOLO26 --layout CHANNELS --output .deps/latency-result --device-only
```

## ⚠️ 平台边界

当前 Android 推理路径绑定 MediaTek Neuron/LiteRT NPU runtime，模型加载会验证 MediaTek 编译插件，并且不会静默切换 CPU。高通平台需要自行接入对应版本的 **QNN Delegate**，适配输入输出张量、量化参数、图编译和 YOLO 后处理验证；本项目的 Neuron 测速数据不适用于高通设备。

## 📄 许可证

YOLOv5、Ultralytics/YOLO26 相关代码和模型遵循各自上游许可证，见 `model-exporter/LICENSE-*.txt`。发布转换后的模型时，请保留源模型许可证、来源和训练数据说明。
