# YOLO26n 320×320 · FYDNN 0.6.1

本版使用官方 YOLO26n COCO 权重，输入固定 320×320、80 类。通过显式调整两个 Attention 分支合并节点的布局，修复本次 MT6993 / Neuron 9.0.11 上原始图首次执行失败的问题。FP32 和 FP16 权重模型已通过一张真实图片的 NPU 执行、输出比对及测速；最终 INT8 的真机验证尚未完成。

## 安装与导入

1. 安装 `FYDNN-0.6.1-arm64.apk`，本版本已在测试手机安装成功。
2. 优先导入 `yolo26n_320_4d_norm_channels_mergefix_fp32io_fp16weights.tflite`。旧版同尺寸模型没有本次修正。
3. 选择 **YOLO26（密集输出 + NMS）**、**CHANNELS**、**80 类**、**归一化坐标**。保留文件名时应用自动选择版本、布局和归一化选项。
4. 点击“编译到联发科 NPU”，等待首次执行自检通过，再选择图片检测，或进入摄像头/OBS。

输入为 `[1,320,320,3]`，输出为 `[1,1,84,2100]`。84 通道为归一化 `cx,cy,w,h` 和 80 类概率，不含 YOLOv5 objectness。应用执行按类别 NMS 并还原画面坐标。FP16 指权重存储精度，输入输出为 FLOAT32，实际硬件计算精度由 Neuron 决定。

最终 INT8 放在 `实验模型-待真机验证/`。已完成完整整数逐张量量化和主机精度检查；测试在 INT8 阶段 ADB 断连，尚无该文件的 NPU 执行、输出一致性或速度通过记录。早期其他 INT8 实验失败的结果不等于最终文件的测试结论。

## 真机结果

2026-10-28，vivo V2548A / MT6993，Neuron 9.0.11，LiteRT 2.3.0 MediaTek NPU-only。主机 TFLite 参考实现使用从手机读取的完全相同输入，比较原始输出。

| 修正模型 | NPU 编译/执行/检测 | 后端平均 / P50 / P95 | 图片端到端平均 / P50 / P95 |
|---|---|---|---|
| FP32 存储 | 通过 | 2.316 / 2.284 / 3.076 ms | 15.417 / 13.489 / 19.690 ms |
| FP16 权重 | 通过 | 3.509 / 3.482 / 4.417 ms | 18.506 / 17.999 / 21.901 ms |
| INT8 | 未完成，测试期间 ADB 断连 | 未测得 | 未测得 |

各已通过模型预热 5 次、统计 50 次。后端包含调用、NPU 完成及输出读回，排除预处理/NMS；端到端另含图片预处理和 NMS，排除采集、网络与显示。FP16 后端倒数约 285 次/秒，图片端到端倒数约 54 次/秒，均不是摄像头实测 FPS。测试为顺序单轮测量，手机功耗、温度、时钟未受控，不据此认定 FP32 普遍更快。

对 COCO128 `000000000009.jpg`，FP16 对同模型主机参考的坐标最大/平均误差为 1.618 / 0.058 像素，类别分数最大误差 0.00358。FP32 为 2.099 / 0.080 像素、0.00726。只覆盖这一张真实图片，不代表完整数据集 NPU 精度验证。散列、原始输入/输出和设备报告见 `验证报告/`。

## 主机精度

| 模型 | 文件大小 | mAP50–95 | AP50 |
|---|---:|---:|---:|
| 原始 ONNX | 9,758,237 字节 | 48.88% | 64.62% |
| 修正 FP32 | 9,760,084 字节 | 48.88% | 64.62% |
| 修正 FP16 权重 | 4,953,316 字节（4.72 MiB） | 48.62% | 64.51% |
| 修正 INT8 | 2,563,200 字节（2.44 MiB） | 43.01% | 62.75% |

评估为 COCO128 固定 32 张图，与 INT8 的 96 张校准图互斥，不是官方 COCO val2017 成绩。INT8 比原模型下降约 5.87 个百分点。INT8 输入输出均为 INT8，浮点张量数、逐通道量化张量数均为零；模型运行时张量最高四维。主机精度与 NPU 输出比对是独立检查。

## 根因定位与修正

原始 320 FP32、FP16、INT8 都发生 Neuron 首次执行失败，224 FP32 诊断版也失败；YOLOv5n 224 INT8 控制组通过。独立矩阵乘法、Softmax、完整 Attention 分支和位置编码分支分别通过，首次失败边界位于 Attention 与位置编码分支合并处。因此不应把问题归结为所有 BATCH_MATMUL 均不支持。

最终修正两个合并节点：NHWC 分支显式转成 `[1,128,100]`，相加后恢复 NHWC。计算与权重保持不变，随机输入全图主机最大误差约 0.0000011。普通 1×1 卷积替换虽能执行但输出偏差过大，已排除；本包只将最终布局修正的 FP32/FP16 列为真机通过。这是当前模型与驱动组合的兼容处理，尚未证明厂商驱动内部具体缺陷。

FYDNN 0.6.1 在加载时执行 NPU 自检并记录加载失败原因，分别报告模型未就绪和未选择图片，保持 NPU-only 执行。图片、摄像头、OBS 共用推理及画框路径；本次验证使用静态图片，未重测摄像头 120 FPS 或 OBS 收流性能。

## 复现与构建

环境：Ultralytics 8.4.190、PyTorch 2.9.1+cpu、ONNX opset 17、onnx2tf 1.26.3、TensorFlow 2.19.0。先用 `tools/export_yolo26.py` 和 `tools/convert_model.py` 生成原始导出并保留 SavedModel，再运行：

```powershell
.toolchain/model-env/Scripts/python.exe tools/rewrite_attention_merge.py --mode nchw-add --tag yolo26-nchw-add-saved
.toolchain/model-env/Scripts/python.exe tools/export_yolo26_mergefix.py
.toolchain/model-env/Scripts/python.exe tools/evaluate_yolo.py --models models/yolo26n-320-neuronfix-v061 --dataset .deps/yolov5/coco128 --onnx models/yolo26n-320-neuronfix-v061/yolo26n_320.onnx --size 320 --family 26
.toolchain/model-env/Scripts/python.exe tools/validate_yolo26_device.py --models models/yolo26n-320-neuronfix-v061 --report .deps/yolo26-neuronfix-device --benchmark
```

原始权重、ONNX、应用/脚本源码、AGPL-3.0 许可证和 SHA256 清单随包提供。主应用不依赖测试组件 APK；真机测试使用已安装的测试组件与 0.6.1 主应用。应用使用 JDK 17、Gradle 8.13、Android SDK Platform 36 / Build Tools 36.0.0。37 项主机测试通过；lint XML 无 Error/Fatal，但日志有 LiteRT Kotlin metadata 2.4 与分析器 2.1 兼容诊断，部分库分析可能受影响。源码包不含本机工具链与缓存，首次构建需下载声明的依赖。
