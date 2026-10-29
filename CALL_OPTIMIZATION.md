# FYDNN 0.5.1：LiteRT 调用层优化候选版

本版已完成离线构建和交付打包，主机 31 项测试通过。安装过程中手机断开 USB，随后用户选择离线交付；尚未完成手机安装、原生调用验证及性能 A/B 测试。0.5.0 已实测版本仍保留在原交付目录。

交付包包含主 APK、测试组件及 v5n 224 / v5s 320 的 FP16、per-tensor INT8 四版模型。日常使用只需安装 `FYDNN-0.5.1-arm64.apk`；`device-tests.apk` 供后续电脑测速使用。模型选择 YOLOv5、行布局、80 类并勾选归一化坐标，再使用图片、摄像头或 OBS/UDP 入口。OBS 的电脑发送目标为手机当前局域网 IPv4，默认 UDP 端口 7890；IP 变化后使用新地址。

## 改动与依据

实际使用的 LiteRT 2.3.0 AAR 字节码显示，`CompiledModel.run(List, List)` 每次会遍历 TensorBuffer，建立两个临时列表、装箱 Long，再转换为两个 long[]，并通过默认签名名称提交。本项目固定单输入、单输出、签名索引 0，缓冲区在模型生命期内复用，因此可在初始化时整理这些句柄。

新增 `PreparedNpuCall`，绑定同一份已安装 LiteRT 库的 `nativeRun(long, int, long[], long[])`，创建时保存模型句柄及输入输出句柄数组，用 MethodHandle 的精确签名直接调用已有 JNI 实现。图片、摄像头及 OBS 共用的 NpuDetector 均使用这条提交路径。模型创建仍使用现有 NPU-only 工厂，所有错误原样上报。

该调用依赖固定的 LiteRT 2.3.0 内部 ABI，主机测试核对了实际依赖中的 JNI 方法签名、同一线程的句柄复用、关闭后禁止调用及错误传播。后续升级 LiteRT 必须重新验证。它复用现成 LiteRT 推理实现，没有自行实现推理引擎或驱动。

## 性能边界

减少的是 Java/Kotlin 包装层的临时分配与签名查找，NPU 计算图本身没有加速承诺。MethodHandle 在 Android ART 上的实际收益也需要测量，测量不增益时不应将此版本认定为提速版。

LiteRT 2.3.0 Java TensorBuffer 仅提供数组读写，没有公开的输出缓冲区映射或 read-into 接口。本版仍执行原有输入写入及完整输出读回，不称为零拷贝。FP16 权重模型的输出是 FLOAT32：v5n224 每帧输出数组约 1.00 MiB、v5s320 约 2.04 MiB；INT8 分别约 0.25、0.51 MiB。这部分数组分配、数据拷贝、JPEG 解码及后处理仍会影响整体帧率。

官方 v2.3.0 MediaTek C 选项定义了性能模式，但本版本 Java 工厂未暴露 MediaTek 选项。此版没有虚构或强行塞入不支持的 Turbo/异步/直接硬件缓冲区参数，也没有将编译缓存当作每帧推理加速。

## 真机验收与测速

手机开启 USB 调试并保持解锁，在源码根目录执行：

```powershell
& '.toolchain/model-env/Scripts/python.exe' tools/benchmark_call_layer.py --install
```

也可只测一个模型：

```powershell
& '.toolchain/model-env/Scripts/python.exe' tools/benchmark_call_layer.py --install --models n224-int8
```

脚本使用独立的 callbench 文件，不覆盖用户已选的 detector.tflite。每个模型在同一 CompiledModel、同一输入和输出 TensorBuffer 上，先各预热 20 次，再交替顺序执行 200 对原公开接口与新接口。测速包含执行完成及读回，另检查两条调用路径的输出是否逐元素完全一致；失败会停止，保留错误报告。报告为平均值、中位数及 P95，位置为 `.deps/call-layer-device`。脚本并不会自动根据测速结果切换执行路径。

测试结果尚未生成；请勿把 0.5.0 的 NPU 耗时或 UDP 回环通过记录当作 0.5.1 的测试结果。持续 OBS/Wi-Fi 帧率仍需另测。模型直接使用上一版交付包中的 FP16 / per-tensor INT8 四维版本。

参考上游：[LiteRT v2.3.0 Model.kt](https://github.com/google-ai-edge/LiteRT/blob/v2.3.0/litert/kotlin/src/main/kotlin/com/google/ai/edge/litert/Model.kt)、[MediaTek 选项](https://github.com/google-ai-edge/LiteRT/blob/v2.3.0/litert/c/options/litert_mediatek_options.h)。
