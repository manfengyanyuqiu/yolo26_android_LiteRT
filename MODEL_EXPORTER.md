# 模型转换工具

`model-exporter/` 提供 YOLOv5、YOLOv8 和 YOLO26 的 PT/ONNX 到 TFLite 转换流程。输出模型经过主机张量检查、单图输出对照和 SHA256 清单生成，可用于 Android 端的 MediaTek Neuron 编译测试。

## 支持范围

| 系列 | 输入 | 输出布局 | 输出内容 |
| --- | --- | --- | --- |
| YOLOv5 n/s | v7 风格解码 ONNX | ROWS | 4 个框坐标、objectness、类别概率 |
| YOLOv8 n/s | PT 或密集 ONNX | CHANNELS | 4 个框坐标、类别概率 |
| YOLO26 n/s | PT 或密集 ONNX | CHANNELS | 4 个框坐标、类别概率 |

支持固定尺寸、batch=1、RGB `[0,1]` 输入和不含内置 NMS 的检测输出。分割、姿态、OBB、动态输入、内置 NMS 和 TopK 输出不在转换范围内。

## 输出精度

- FP32：FP32 权重、FLOAT32 输入输出。
- FP16：FP16 权重、FLOAT32 输入输出；实际算术精度由 Neuron 驱动决定。
- INT8：全整数输入输出，使用校准图片计算量化参数。

INT8 转换至少需要 32 张具有代表性的 JPG/PNG 校准图片。建议使用与实际场景相近的数据，转换后再在目标手机上检查输出和速度。

## 使用方式

1. 复制 `model-exporter/exporter-runtime.example.json` 为 `exporter-runtime.json`。
2. 填写导出 Python、转换 Python、校准图片目录和纯英文临时工作目录。
3. 运行 `model-exporter/launch_exporter.cmd`，或直接调用 `tools/model_exporter.py`。
4. 选择模型系列、n/s 规模、固定输入尺寸和 `fp16`/`int8`/`fp32` 输出。
5. 检查输出目录中的 `conversion-validation.json`、`final-parity.json`、`tensor-report.json` 和 `SHA256.json`。

示例：

```powershell
python tools/model_exporter.py `
  --family v8 --model n --size 320 `
  --weights path/to/yolov8n.pt `
  --formats fp16 int8 `
  --calibration-dir path/to/calibration/images `
  --output path/to/exported-model
```

主机转换通过只说明 TFLite 文件结构和主机参考输出通过；MediaTek Neuron 编译、真机兼容性和速度需要在目标设备上单独验证。高通平台需要自行接入 QNN Delegate。
