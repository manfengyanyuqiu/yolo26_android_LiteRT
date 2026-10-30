# 模型转换工具

该目录提供 YOLOv5、YOLOv8、YOLO26 的 ONNX/PT 到校验后 TFLite 转换流程，复用 Ultralytics、onnx2tf、TensorFlow Lite 和 ONNX Runtime。

复制 `exporter-runtime.example.json` 为 `exporter-runtime.json`，填写本机 Python 环境；`calibration_dir` 至少包含 32 张 JPG/PNG，`work_dir` 应使用纯英文路径。

支持 v5 v7 风格解码 ONNX（ROWS）、v8/YOLO26 PT 或密集 ONNX（CHANNELS），以及 FP32、FP16 权重和 INT8 全整数导出。导出结果包含阶段日志、张量报告、精度对照和 SHA256 清单。导出成功只代表主机转换校验通过，手机 Neuron 编译和真机测速需要单独执行。

## Neuron INT8 分类头处理

图形界面和 `model_exporter.py` 流程自动对匹配拓扑的 YOLOv8/YOLO26 INT8 分类头执行 Sigmoid 布局变换。Sigmoid 是逐元素操作，可移至三个 NHWC 类别卷积输出，随后再转置、重排和拼接；模型的 INT8 输入输出、量化参数与检测解码约定保持一致。报告中的 `neuron_class_sigmoid` 记录是否应用。该处理仅针对独立分类头。YOLOv5 和框/类别先合并再切片的拓扑不匹配，保持原图并在报告中明确注明不适用；发现不支持或共享分支时明确报错。

已有全整数模型可单独处理，输出使用新文件名：

```powershell
python tools/fix_neuron_int8.py path/to/original.tflite --output path/to/headsigmoid_int8.tflite
```

底层 `convert_model.py` 可使用 `--npu-int8-head-sigmoid` 开启同一处理。该变换针对分类头，不替代校准和目标设备精度验证；转换成功仍需真机 Neuron 编译、数值及画框检查。
