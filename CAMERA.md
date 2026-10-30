# Camera2 实时检测

应用使用手机后置 Camera2 摄像头采集画面，通过 OpenGL ES 预览、RGBA 读回、模型预处理、MediaTek Neuron NPU 推理和检测框叠加完成实时检测。

## 采集模式

应用优先选择设备公开的 constrained high-speed 固定 120 FPS 模式，并使用该模式允许的最小采集分辨率。设备没有公开固定 120 FPS 模式、会话配置失败或采集失败时，界面直接显示错误和设备公开的帧率范围，不静默改成其他帧率。

摄像头采集分辨率与模型输入尺寸相互独立。相机通常输出 1280x720 或设备公开的其他尺寸，预处理会按模型输入尺寸执行 letterbox、颜色转换和量化；当前发布模型主要使用 320x320 输入。

## 处理链路

```text
Camera2
  -> SurfaceTexture / OpenGL ES 外部纹理
  -> 方向变换与 letterbox
  -> RGBA 读回与 RGB/INT8 输入打包
  -> LiteRT MediaTek Neuron NPU
  -> YOLOv5/v8/26 解码与 NMS
  -> 预览叠加检测框
```

推理界面显示单次 NPU 推理和同步输出读回耗时，并可同时显示采集帧率。应用丢弃处理不及实时节奏的旧帧，只保留最新完成结果，避免队列持续堆积。

## 使用条件

首次进入摄像头页面需要授予相机权限。设备需要 Android 12（API 31）或更高版本、`arm64-v8a` ABI 和可用的 MediaTek Neuron/LiteRT 驱动。没有可用的固定 120 FPS 能力时，请使用图片检测入口进行模型功能验证。
