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
| `ModelImportOptions.java` | 从输出张量识别布局，并解析发布模型的系列和坐标约定 |
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

### 📦 安装与模型导入

在 [Releases](https://github.com/manfengyanyuqiu/yolo26_android_LiteRT/releases/latest) 下载 APK，或下载包含 APK 与 12 个 320×320 模型的完整 ZIP。模型覆盖 YOLOv5n/s、YOLOv8n/s、YOLO26n/s，每种各有 FP16 权重 / FLOAT32 I/O 和 INT8 两版。

1. 安装 APK，点击“选择 TFLite 模型”，选择解压后的 `.tflite` 文件。
2. 应用从文件名识别 YOLO 系列，从实际输出张量识别 `[1,N,C]` 或 `[1,C,N]` 布局；输入尺寸自动读取，无需手填。包内模型类别数均为 **80**。
3. 包内模型使用归一化坐标，导入后自动勾选。文件名中的 `_norm_` 表示该约定，旧发布包的 12 个简化文件名也支持自动识别。自定义模型需要按实际导出设置核对系列、类别数和归一化坐标；张量形状本身无法说明坐标是否归一化。
4. 点击“编译到联发科 NPU”。模型就绪后，选择图片并点击“检测当前图片”，或进入摄像头检测。
5. 测速时使用同一张图片，点击“测速：5 次预热 + 50 次推理”，比较平均耗时（ms）。首次编译耗时不属于推理耗时。

模型参数校验错误与 NPU 编译或首次执行错误分别显示。若输出为 `[1,1,84,2100]`，它对应 v8/26 密集检测的 `[1,C,N]` 布局，不应选择 v5 的 `[1,N,C]` 布局。界面默认参数也不能代替模型的实际导出约定。

APK 0.6.5 的导入检查已在 vivo V2548A / MT6993 / Android 17 上验证全部 12 个模型：通过文件内容 URI 导入，检查自动参数，然后点击编译和图片检测按钮。新旧两套文件名共 24 项通过，详见 [导入与编译验证结果](benchmarks/model-import-0.6.5.json)。该记录对应导入检查时的模型 SHA256，不替代当前修正模型的数值验证；原生文件选择器对话框本身未自动化操作。当前三个 INT8 模型的重新导入、编译、画框和测速记录见下方真机验证报告。

## 🔄 模型转换

模型转换工具位于 [`model-exporter/`](model-exporter/)。它支持 YOLOv5、YOLOv8、YOLO26 的 PT 或 ONNX 输入，并输出 FP32、FP16 权重和 INT8 全整数 TFLite。输出尺寸必须是 32 的倍数，默认使用 320x320；输出模型不应包含内置 NMS。

先复制运行时配置并填写本机 Python 环境：

```powershell
Copy-Item model-exporter/exporter-runtime.example.json model-exporter/exporter-runtime.json
.\model-exporter\launch_exporter.cmd
```

转换工具会自动将匹配拓扑的 v8/26 INT8 分类 Sigmoid 移到各 NHWC 检测头，保持量化参数和输入输出契约；不匹配时会在报告中注明，支持路径出现冲突时直接报错。

INT8 导出至少需要 32 张具有代表性的 JPG/PNG 校准图片。转换流程会生成阶段日志、张量报告、主机精度对照和 SHA256 清单。主机转换通过后，仍需在手机上单独执行 Neuron 编译和真机验证。

更多参数和 ONNX 布局说明见 [`model-exporter/README.md`](model-exporter/README.md) 与 [`MODEL_EXPORTER.md`](MODEL_EXPORTER.md)。

## ⚡ 天玑 9500 真机测速

测试日期：2026-10-30。设备：vivo V2548A / MT6993 / Android 17。后端：LiteRT 2.3.0 MediaTek Neuron，NPU-only。全部模型统一使用 **320×320** 输入，同一张 640×480 图片，每项指标分别预热 5 次、采样 50 次，表中为算术平均耗时。

YOLOv8n、YOLO26n/s 的 INT8 行使用本次 `_headsigmoid_int8` 修正模型实测结果，其余 9 行来自同日较早场次。各行对应实际模型 SHA256；本次三个结果取自界面，保留到 0.01 ms。

| 模型 | 精度 | 输入尺寸（px） | NPU 执行及读回平均耗时（ms） | 图片处理平均耗时（ms） |
| --- | --- | --- | ---: | ---: |
| YOLOv5n | FP16 权重 / FLOAT32 I/O | 320×320 | 4.94 | 6.00 |
| YOLOv5n | INT8 | 320×320 | 1.97 | 3.57 |
| YOLOv5s | FP16 权重 / FLOAT32 I/O | 320×320 | 6.19 | 7.33 |
| YOLOv5s | INT8 | 320×320 | 3.09 | 4.72 |
| YOLOv8n | FP16 权重 / FLOAT32 I/O | 320×320 | 3.07 | 5.64 |
| YOLOv8n | INT8 | 320×320 | 1.28 | 3.00 |
| YOLOv8s | FP16 权重 / FLOAT32 I/O | 320×320 | 6.15 | 8.18 |
| YOLOv8s | INT8 | 320×320 | 2.34 | 6.53 |
| YOLO26n | FP16 权重 / FLOAT32 I/O | 320×320 | 4.19 | 6.69 |
| YOLO26n | INT8 | 320×320 | 1.49 | 3.93 |
| YOLO26s | FP16 权重 / FLOAT32 I/O | 320×320 | 5.93 | 8.38 |
| YOLO26s | INT8 | 320×320 | 1.63 | 4.01 |

NPU 执行及读回：从提交已准备的 NPU 调用开始，到同步读取输出完成。图片处理：从已解码的 Bitmap 开始，包含 letterbox、RGB/量化输入打包、NPU 执行、同步读回和 YOLO 解码/NMS。两项均不包含模型编译、JPEG 解码、摄像头采集或屏幕显示。

12 个模型均完成 NPU 编译和执行。FP16 指权重存储为 FP16、输入输出为 FLOAT32，硬件算术精度由 Neuron 驱动决定。当前 YOLOv8n、YOLO26n/s INT8 模型将分类 Sigmoid 放在各 NHWC 检测头中，避免 Neuron 对合并后分类张量执行时产生密集误框。三个模型各通过 32 张图片的主机逐位一致性检查、8 张图片的 NPU 对照，本次测试图分别检出 5、4、4 个目标。主机与 NPU 仍有数值差异，这些检查不等于完整 mAP 精度评估；部署前需用实际数据集验证。各模型依次测试，保留设备正常调频；温度、后台负载、驱动和图片内容均会影响耗时。详见 [INT8 模型验证与本次真机结果](benchmarks/2026-10-30-int8-headfix.json)。

[机器可读结果、模型/图片 SHA256 与 P50/P95](benchmarks/2026-10-30-320.json)。高通设备需要自行接入 QNN Delegate，以上数据仅适用于本次 MediaTek 设备测试。

```powershell
python tools/benchmark_latency.py --model path/to/model.tflite `
  --family YOLO26 --layout CHANNELS --output .deps/latency-result --device-only
```

## 🎯 与 ONNX Runtime 的精度对照

测试日期：2026-10-30。使用 v0.6.6 发布包的 12 个 **320×320** 模型，在 vivo V2548A / MT6993 / Android 17 上执行 Neuron NPU 推理，与对应原始 ONNX 的 ONNX Runtime CPU 结果比较。数据为 COCO128 中未参与 INT8 校准的 **32 张图片、216 个标注框**，12 个模型共完成 **384 次真机图片推理**。

下表指标为 **mAP50–95（%）**；最后一列为 Neuron INT8 相对 ONNX Runtime 的百分点变化。

| 模型 | 原始 ONNX Runtime | Neuron FP16 | Neuron INT8 | INT8 相对基线（百分点） |
| --- | ---: | ---: | ---: | ---: |
| YOLOv5n | 37.39 | 38.04 | 30.76 | -6.63 |
| YOLOv5s | 49.83 | 49.60 | 44.20 | -5.62 |
| YOLOv8n | 48.53 | 48.48 | 42.36 | -6.18 |
| YOLOv8s | 55.25 | 54.81 | 48.17 | -7.07 |
| YOLO26n | 49.36 | 49.47 | 43.44 | -5.91 |
| YOLO26s | 57.62 | 57.64 | 46.26 | -11.37 |

FP16 在这批图片上与 ONNX Runtime 基线接近，差值约为 −0.43～+0.65 个百分点；正向小差值不代表真实精度提升。INT8 下降约 5.62～11.37 个百分点，主机 TFLite 中间对照表明，大部分损失在转换/量化阶段已经出现。

**YOLOv8s INT8 另有置信度异常**：固定置信度阈值 0.25、匹配 IoU 0.50 时，micro Recall 从 ONNX Runtime 的 **51.85% 降至 28.24%**。同一批 32 张图的 NPU 重复输出逐字节一致，问题可复现；不能仅凭总体 AP 判断部署效果。

所有后端使用相同的 OpenCV letterbox 基础输入和统一后处理：类别内 NMS IoU 0.45，AP 计算的置信度下限为 0.001。该测试隔离比较模型与后端，未包含 Android Bitmap/摄像头预处理和 Java 解码器的逐项一致性验证。**32 张图的小样本结果不是完整 COCO val2017 成绩**；COCO128 来自训练子集，可能与源权重训练数据重叠。v5 原始 ONNX 基线为 FLOAT16，v8/26 为 FLOAT32；Neuron FP16 指 FP16 权重、FLOAT32 I/O。

[完整精度报告：主机中间对照、AP50、Precision/Recall 与限制](ACCURACY_BENCHMARK.md) · [原始指标、模型/图片哈希及数值误差](benchmarks/2026-10-30-accuracy-320.json)

## ⚠️ 平台边界

当前 Android 推理路径绑定 MediaTek Neuron/LiteRT NPU runtime，模型加载会验证 MediaTek 编译插件，并且不会静默切换 CPU。高通平台需要自行接入对应版本的 **QNN Delegate**，适配输入输出张量、量化参数、图编译和 YOLO 后处理验证；本项目的 Neuron 测速数据不适用于高通设备。

## 📄 许可证

YOLOv5、Ultralytics/YOLO26 相关代码和模型遵循各自上游许可证，见 `model-exporter/LICENSE-*.txt`。发布转换后的模型时，请保留源模型许可证、来源和训练数据说明。
