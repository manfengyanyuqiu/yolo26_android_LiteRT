# 天玑 9500 真机速度

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
