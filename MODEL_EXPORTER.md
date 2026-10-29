# FYDNN 模型导出工具 0.6.3

2026-10-28。双击 launch_exporter.cmd 打开界面，选择系列、n/s、尺寸、源模型和导出精度。手机继续导入 .tflite；.onnx 是电脑端转换的源文件。

## 支持范围

| 系列 | 源模型 | 输出布局 | 检测概率 |
| --- | --- | --- | --- |
| YOLOv5 n/s | v7 风格 /model.24/ 检测头 .onnx | ROWS，85 通道（80 类） | objectness × 类别概率 |
| YOLOv8 n/s | 官方/对应结构 .pt 或解码后的 .onnx | CHANNELS，84 通道（80 类） | 类别概率 |
| YOLO26 n/s | 官方/对应结构 .pt 或密集输出 .onnx | CHANNELS，84 通道（80 类） | 类别概率 |

v8 / 26 的源模型留空时下载对应官方 PT。v5 请先用原 YOLOv5 导出工具生成 ONNX；本工具不直接加载旧版 v5 PT。仅支持目标检测、batch=1、RGB [0,1]、NCHW 源输入和未内置 NMS 的像素 cxcywh 解码输出；分割、姿态、OBB、内置 NMS/TopK 输出不在本版本支持范围。自定义类别数从输出推断，须保留对应解码头结构；配套 COCO 评估只接受 80 类和 COCO 类别顺序。已实测模型见 verification.json。

输入尺寸为 32 的倍数，默认 320×320。固定 ONNX 的输入尺寸须与选择一致。动态 ONNX 使用上游 onnxsim 固定输入形状，通过 ONNXRuntime 比对原动态图和固定图输出，再检查系列对应解码头结构；当前动态导入已实测 v8s。原动态图保存为 .original.onnx，固定图另存，源文件保持原样。

## 精度与使用

1. FP32：FLOAT32 输入输出。FP16：FP16 权重存储、FLOAT32 输入输出；实际硬件计算精度由驱动决定。INT8：全整数输入输出、逐张量权重，导出时检查无浮点张量。
2. INT8 至少需要 32 张真实 JPG/PNG 校准/评估图。按文件名排序后每四张留一张评估，其余校准；默认 COCO128 为 96/32、互不重叠。优先选择实际检测画面的代表图。
3. 仅导出浮点模型时可以只选一张验证图片。勾选 COCO 评估需要 images/train2017 及配套 labels/train2017。自有校准目录可关闭此项。
4. 输出目录须为新目录或空目录。日志按阶段实时写入，任何转换或校验失败都会停止。
5. 在 FYDNN 选择对应 v5/v8/26 系列、表中的布局、实际类别数，并开启归一化坐标。保留原文件名时，现有 APK 按文件名选择系列、布局和归一化选项。

最终模型使用 NHWC 输入 [1,H,W,3]，输出保持四维。320 输入、80 类时，v5 输出 [1,1,6300,85]，v8/26 输出 [1,1,84,2100]。YOLO26 另执行 Attention 合并布局修正。

## 校验与报告

复用 Ultralytics / onnx2tf / TensorFlow / ONNXRuntime：源图检查及实际执行 → FP32 中间图转换比对 → 系列对应布局修正 → 所选精度导出 → 全图最大张量维度检查 → 最终文件单图比对 → 可选数据集评估。

中间 FP32 转换要求坐标最大误差 ≤0.05 像素、概率误差 ≤0.001。源 ONNX 为 FP16 时，保留权重数值并提升图的运算类型，使用提升后的 FP32 图检验转换本身；最终模型仍与原 FP16 ONNX 比较。最终 FP16 权重或 FP16 源模型比较要求概率误差 ≤0.05、参考检测分数 ≥0.1 的候选框坐标误差 ≤2 像素。v5 检测分数包含 objectness。报告同时记录全部候选框的误差。

INT8 执行检查记为 HOST_EXECUTION_PASS，精度看 accuracy-report.json。开启评估时浮点 mAP50–95 下降超过 2 个百分点会停止。默认评估只有 32 张图，是小样本参考，并非官方 COCO val2017。

输出含原始/固定 ONNX、PT（若使用）、export-job.json、逐阶段日志、SHA256.json、tensor-report.json、conversion-validation.json、final-parity.json、INT8 校准报告及可选 accuracy-report.json。conversion-validation.json 的哈希属于工作目录中的中间产物；最终文件哈希以 SHA256.json 和 final-parity.json 为准。PT 的 PyTorch/ONNX 比对结果保留在 .source.json 的 upstream_export_report 中。

“导出完成”表示主机转换与所选检查通过。新模型的手机 NPU 编译、执行与耗时须另行真机测试。FP16 指权重存储，驱动控制实际运算精度；INT8 的执行成功不等同于精度与浮点模型一致。

## 本机环境

此版本是本机 Python 图形/命令行工具，不内置大型 Python 依赖。交付目录 exporter-runtime.json 指向现有工程的两个 Python 环境及校准目录，移动原工程后需更新路径。Tkinter 提供图形界面。

Windows TensorFlow 的 SavedModel 工作目录使用纯英文路径：交付配置中的 work_dir 指向原工程 .deps/export-jobs，也可通过 --work-dir 指定。未配置时使用系统临时目录下 fydnn-export-jobs；路径含非 ASCII 字符会明确报错并要求指定工作目录。工具目录、最终输出目录和验证图片可使用中文路径，TFLite 文件通过 Python 字节读取进入解释器。工作目录在 export-job.json 中记录，保留用于排查失败与复核中间图。

现有环境：Python 3.12、Ultralytics 8.4.190、PyTorch 2.9.1+cpu、onnx2tf 1.26.3、TensorFlow 2.19.0。requirements-export.txt 和 requirements-convert.txt 分别对应导出、转换环境。

命令行示例（在原工程目录运行）：

    .toolchain/model-env/Scripts/python.exe tools/model_exporter.py --family v5 --model s --size 320 --onnx models/yolov5s-320-mediatek/yolov5s_320.onnx --formats fp16 int8 fp32 --output exports/v5s-320-new --evaluate

    .toolchain/model-env/Scripts/python.exe tools/model_exporter.py --family v8 --model s --size 320 --onnx ../yolov8s.onnx --formats fp16 int8 fp32 --output exports/v8s-320-new --evaluate

    .toolchain/model-env/Scripts/python.exe tools/model_exporter.py --family v8 --model n --size 320 --weights models/exporter-v063-v8n-source/yolov8n.pt --formats fp16 int8 fp32 --output exports/v8n-320-new --evaluate

    .toolchain/model-env/Scripts/python.exe tools/model_exporter.py --family 26 --model s --size 320 --weights models/yolo26s-320-source/yolo26s.pt --formats fp16 --output exports/26s-320-new

相关模型及 Ultralytics 分发遵循随包 AGPL-3.0 / YOLOv5 v7 GPL-3.0 许可证。
