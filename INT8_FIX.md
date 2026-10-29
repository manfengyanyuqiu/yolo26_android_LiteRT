# INT8 编译修复 · FYDNN 0.4.1

2026-10-27 已在 vivo X300s（V2548A / MT6993 / Android 17 / Neuron 9.0.11）复现并验证。

## 根因与修改

原 INT8 模型在第一个卷积算子导入时失败。LiteRT 2.3.0 MediaTek 插件日志：

```
operand_map.cc:98 Failed to set symmetric per-channel quant params
operand_map.cc:159 Failed to set general per-channel quant params
```

改用 TensorFlow Lite 转换器的每张量权重量化选项，重新使用 96 张 COCO128 图片校准。两份模型保持 INT8 输入、输出、权重与激活，所有张量最高 4 维，浮点张量为 0，逐通道量化张量为 0。转换工具新增显式参数 `--npu-int8-per-tensor`，不在运行失败后静默换模型。

编译通过后，旧版应用还会发生 `Failed to find input: serving_default_images:0`：TFLite 底层张量名与 SignatureDef 中的输入别名不同。0.4.1 使用 LiteRT 官方 `createInputBuffers(0)` / `createOutputBuffers(0)` 按签名顺序创建缓冲，与同一索引的 `run` 对应。FP16 无显式签名模型也已回归验证。

## 真机验证

使用应用实际 NPU-only 路径，单图检测、5 次预热、50 次采样，两份 INT8 均通过。数值为当次测量，包含同步输出读回；不含相机采集和显示，不等于相机检测 FPS。

| 模型 | INT8 文件大小 | 推理＋同步读回平均 | 图片端到端平均 |
|---|---:|---:|---:|
| YOLOv5n 224 | 1,965,080 字节 | 1.005 ms | 2.494 ms |
| YOLOv5s 320 | 7,350,328 字节 | 2.318 ms | 4.458 ms |

旧 FP16 权重 224 与 320 两版在 0.4.1 上也通过完整图片推理。21 项主机测试、APK 构建、签名验证通过，lint 无错误；lint 分析器仍有上游 Kotlin 元数据版本诊断。

## 精度与边界

留出 COCO128 中另外 32 张图片在主机 CPU 上评估，未用于校准：

| 模型 | FP16 权重版 mAP50–95 | 新每张量 INT8 mAP50–95 |
|---|---:|---:|
| YOLOv5n 224 | 29.11% | 26.65% |
| YOLOv5s 320 | 49.65% | 43.92% |

YOLOv5s 原逐通道 INT8 为 46.35%，新兼容版低约 2.42 个百分点；这是兼容当前 Neuron 导入接口的精度代价。以上为小规模转换对比，不是官方 COCO val2017 或真实业务数据精度。

另用设备实际输入逐字节回传，输入相同的主机 TFLite 与 NPU 输出并非完全一致：224 平均差 0.288 个量化单位、最大 34；320 平均差 0.317 个量化单位、最大 74。这项单图比较只检查读回和运行，不证明 NPU 与 CPU 数值等价，也不代表整套数据的设备精度。明细保存在交付包验证报告。

## 使用

安装 0.4.1 后重新导入文件名带 `_int8_per_tensor` 的模型。选 YOLOv5 / ROWS / 80 类 / 归一化坐标。不要继续导入旧交付包的逐通道 INT8 文件。FP16 权重模型可继续使用。

相机模式保留，但本次只验证图片编译、推理和测速，相机 120 FPS 尚未验证。
