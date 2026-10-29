# 天玑 9500 / YOLOv5n 224 Android 交付

**当前版本 0.4.0：屏幕截取已替换为摄像头 120 FPS 请求、实时预览与画框。使用方法、采集管线、计时边界和真机待验证事项见 [CAMERA.md](CAMERA.md)。下文 0.3.0 部分记录两份模型的来源和精度。**

版本 0.3.0，ARM64，Android 12 及以上。APK 在 app/build/outputs/apk/debug/app-debug.apk。整理后的安装包和两版模型在上级目录“天玑9500-YOLOv5n224”。

## 安装与配置

1. 安装 FYDNN-0.3.0-arm64.apk。
2. 将两份 .tflite 放到手机 Download，应用中选择其中一份。
3. 设置 **YOLOv5、ROWS [1,N,C]、80 类、勾选归一化坐标**。导入文件名含 _norm_ 时自动勾选，请保留文件名。
4. 点击“编译到联发科 NPU”，再选择图片检测或测速。
5. 换另一模型，使用同一图片再次测速。每项 5 次预热、50 次计时，显示平均、P50、P95。

输入 RGB、灰色 114 letterbox、除以 255、固定 NHWC [1,224,224,3]。输出 [1,1,3087,85]，依次为归一化中心坐标/宽高、objectness、80 类概率。宽高可能超过 1，后处理映射回原图再裁边。App 置信度阈值 0.25，按类别 NMS IoU 0.45。

## 两版模型

最终模型目录为 models/yolov5n-224-mediatek。本次是官方 **YOLOv5n v7.0 COCO 80 类权重**，不是先前的 YOLOv5s，也不是 .fymf 内的自定义权重。

| 文件 | 字节 | 输入/输出 | 存储和量化 |
| --- | ---: | --- | --- |
| yolov5n_224_4d_norm_fp32io_fp16weights.tflite | 3,828,264 | FLOAT32 / FLOAT32 | FP16 权重 |
| yolov5n_224_4d_norm_int8.tflite | 2,096,048 | INT8 / INT8 | 权重与激活全整数量化，卷积权重按通道 |

FP16 权重文件不等于已证明 NPU 硬件以 FP16 执行；最终内部精度由 MediaTek 编译器和驱动决定。本次没有连接手机，两个新模型的 NPU 编译、检测和速度均待设备验证。

INT8 文件无浮点张量，输入 scale=0.003921568859368563、zero=-128；输出 scale=0.007096786051988602、zero=-126。App 从模型读取量化参数。

## 推理路径优化

- 沿用 LiteRT 2.3.0 MediaTek NPU-only 路径，编译失败直接报错，不自动切换 CPU。Java 发布包默认加入 CPU 的行为由固定 ABI 的 NpuOnlyModel.java 显式绕开。
- 检测头 H/W 轴等价合并为 rank 4，避开此前真机 Neuron 9 的 padded-dimensions rank-5 错误；全部张量最高 rank 4。检测头保留在模型图中。
- 坐标在检测头合并概率之前归一化，减少 INT8 输出中坐标与概率的数值范围差异。
- 复用 Bitmap、Canvas、Paint、Rect、像素数组、输入数组和 buffer 列表。RGB 归一化与输入量化查表。
- INT8 输出保持 byte[]，只按访问位置反量化，不再每帧创建完整 float[]。224 输出读回为 262,395 字节，FP16 权重版 FLOAT32 输出为 1,049,580 字节。
- objectness 低于可能达到阈值的上限时跳过 80 类扫描，保持阈值边界与按类 NMS。
- 0.3.0 曾使用屏幕中心裁剪；0.4.0 改为 Camera2 / OpenGL 摄像头输入，见 CAMERA.md。

当前 LiteRT Java TensorBuffer.readFloat/readInt8 每次仍返回新数组，本次没有实现零拷贝或自定义 JNI 内存通路。后端测速包含 model.run 和同步输出读回；图片端到端另含预处理、解码，不含采集、画框及显示。

## 校准与精度验证

复用 TFLite 官方全整数 PTQ、onnx2tf 1.26.3 和 pycocotools COCOeval。COCO128 按文件名排序，每第四张留作评估，其余校准：**96 张校准、32 张评估，互不重叠**。这是通用 COCO 校准，不是用户实际场景校准，也不是 COCO val2017 的正式指标。

32 图主机 CPU 参考评估，分数阈值 0.001、按类 NMS IoU 0.45：

| 模型 | mAP50–95 | AP50 |
| --- | ---: | ---: |
| 原始 224 ONNX | 29.09% | 44.45% |
| FP16 权重 TFLite | 29.11% | 44.46% |
| INT8 TFLite | 26.22% | 43.39% |

INT8 相对 FP16 权重版 mAP50–95 下降约 2.90 个百分点。FP16 与原 ONNX 最大坐标误差约 0.626 像素，最大概率误差（含 objectness）约 0.00235。INT8 全候选框最大坐标误差约 146.06 像素、平均约 1.57 像素，最大概率误差约 0.412；量化精度结论使用检测 AP，不把转换成功当成精度一致。

accuracy-report.json、validation.json、tensor-report.json 和 *.quantization.json 记录样本、哈希、误差、张量/算子及量化参数。官方固定 640 ONNX 的静态网格/reshape 改为 224，所有学习参数逐字节保留，来源与更改记录在 yolov5n_224.source.json。校准和评估采用相同的 RGB/letterbox 约定；主机浮点 bilinear 与 Android Bitmap 插值不保证逐像素相等。

## 检查和设备边界

- 0.3.0 的 12 项模型/解码/ABI 测试继续通过；0.4.0 新增 9 项相机管线测试，共 21 项，主 APK 和 instrumentation APK 构建成功。
- Android Lint：0 errors、11 warnings，主要为中文硬编码、ARM64 专用和配置提示。分析器另提示 LiteRT Kotlin metadata 2.4 与其分析器 2.1 的版本差异；Java 编译打包通过。
- 本次无手机，未测 YOLOv5n 224 NPU FPS，未测优化前后真机提速比例。
- 历史 YOLOv5s 640 rank-4 FP16 权重模型已在 vivo X300s / MT6993 / Android 17 / Neuron 9.0.11 成功编译运行，日志显示 MediaTek compiler plugin 和单个 dispatch partition。该结果只证明当时模型，不代表本次新模型已验证。
- App 的 Interpreter 只读元数据，实际推理通过 NPU-only CompiledModel；主机 Python CPU 推理仅用于离线转换和精度检查。

手机接回后可在工程目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File tools/benchmark_224.ps1
```

脚本安装两个 APK，分别加载 FP16/INT8，在同一图上记录后端及图片端到端平均/P50/P95，失败立即停止。报告在 .deps/npu-debug/yolov5n-224。可用 -Serial 10AG4M1TY4001SG 选择设备。

## 复现

在工程目录使用已准备的工具链：

```powershell
$env:JAVA_HOME = (Resolve-Path '.toolchain/jdk-17.0.20.1+1').Path
$env:PATH = "$env:JAVA_HOME\bin;$env:PATH"
$env:GRADLE_USER_HOME = (Resolve-Path '.toolchain/gradle-home').Path
& '.toolchain/gradle-8.13/bin/gradle.bat' --offline testDebugUnitTest assembleDebug assembleDebugAndroidTest lintDebug
```

模型来源缓存 .deps/yolov5/yolov5n.onnx 与 .deps/yolov5/coco128：

```powershell
& '.toolchain/model-env/Scripts/python.exe' tools/prepare_yolov5n.py .deps/yolov5/yolov5n.onnx models/yolov5n-224-mediatek/yolov5n_224.onnx --size 224
& '.toolchain/model-env/Scripts/python.exe' tools/convert_model.py models/yolov5n-224-mediatek/yolov5n_224.onnx --family v5 --name yolov5n_224 --output models/yolov5n-224-mediatek --size 224 --test-image ../fydnn_demo/x64/Release/1.png --mediatek-4d --normalize-coordinates --calibration-dir .deps/yolov5/coco128/images/train2017 --keep-saved-model
& '.toolchain/model-env/Scripts/python.exe' tools/evaluate_yolo.py --models models/yolov5n-224-mediatek --dataset .deps/yolov5/coco128
& '.toolchain/model-env/Scripts/python.exe' tools/inspect_tflite.py models/yolov5n-224-mediatek
```

源码保留在本工程；原 ONNX、Windows 库与 .fymf 保持原样。models 下旧试验不作为本次交付。

来源：YOLOv5 v7.0 https://github.com/ultralytics/yolov5/tree/v7.0 ，权重 https://github.com/ultralytics/yolov5/releases/tag/v7.0 ，数据 https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip 。YOLOv5 v7.0 LICENSE 随交付附上。
