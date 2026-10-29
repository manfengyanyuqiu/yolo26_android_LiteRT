# 模型转换工具

该目录提供 YOLOv5、YOLOv8、YOLO26 的 ONNX/PT 到校验后 TFLite 转换流程，复用 Ultralytics、onnx2tf、TensorFlow Lite 和 ONNX Runtime。

复制 `exporter-runtime.example.json` 为 `exporter-runtime.json`，填写本机 Python 环境；`calibration_dir` 至少包含 32 张 JPG/PNG，`work_dir` 应使用纯英文路径。

支持 v5 v7 风格解码 ONNX（ROWS）、v8/YOLO26 PT 或密集 ONNX（CHANNELS），以及 FP32、FP16 权重和 INT8 全整数导出。导出结果包含阶段日志、张量报告、精度对照和 SHA256 清单。导出成功只代表主机转换校验通过，手机 Neuron 编译和真机测速需要单独执行。
