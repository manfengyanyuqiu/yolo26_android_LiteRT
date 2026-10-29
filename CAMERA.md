# 0.4.0 摄像头实时检测

原屏幕采集入口已替换为摄像头检测。安装新版 APK，选择上一版的 YOLOv5n 224 模型，保持 YOLOv5 / ROWS / 80 类 / 勾选归一化坐标，然后点击“摄像头检测：120 FPS 采集 + 实时画框”。此入口自行编译模型，无需先做图片检测。首次使用允许相机权限。

使用后置相机。按公开 Camera2 能力选择固定 [120,120] FPS：优先 constrained-high-speed 会话及其支持的最小分辨率；只有普通会话公开固定 120 FPS 且对应尺寸的最小帧间隔符合时才使用普通会话。相机采集尺寸通常为 1280×720 或其他设备公开尺寸，模型输入仍为 224×224。单个 SurfaceTexture 输出是 Android 高速会话支持的标准输出类型，通过 createHighSpeedRequestList + setRepeatingBurst 提交。

没有固定 120 FPS 模式、会话配置失败或采集失败时显示实际错误及可见 FPS 范围，不静默降成 30/60 FPS。第三方应用公开能力与系统相机慢动作模式可能不同；没有手机，本次尚未确认 vivo X300s 是否向此应用开放固定 120 FPS。

## 采集、推理与画框

Camera2 → SurfaceTexture / OpenGL 外部纹理 → 按传感器与屏幕方向旋转 → GPU letterbox → 模型尺寸 RGBA 读回 → RGB/INT8 输入打包 → LiteRT MediaTek NPU → 解码、NMS → 预览叠加框。

预览直接采样原相机纹理；只有模型输入从 GPU 读回，224 输入每次为 200,704 字节。复用单个 direct ByteBuffer，不逐帧创建 Bitmap、不将原始 720p/1080p 全分辨率图像拷到 Java。RGBA 打包处理 OpenGL 行倒序，跳过 alpha，保持 RGB 顺序。模型输出框保留在同一旋转/letterbox 坐标系，按预览尺寸缩放画框，不再重复旋转。

推理只有一个在途输入；忙时继续显示新预览，跳过旧帧，不排无限帧队列。完成后再取最新可用帧。画框显示最近一次检测结果，超过 500 ms 的旧结果清除。预览显示请求 120 Hz，由系统和屏幕实际调度。

这表示请求 120 FPS 相机采集、以 NPU 实际处理能力检测最新帧，不承诺每个相机帧都完成推理。界面独立显示：

- 相机回报 FPS：Camera2 SENSOR_TIMESTAMP 回调推算，高速批处理时可能含框架生成的批次内元数据。
- GPU 取帧 FPS：实际消费的不同 SurfaceTexture 时间戳数；受图像队列和显示刷新率影响。
- 检测 FPS：实际完成推理、解码的结果数。
- GPU 读回耗时；NPU run + 输出同步读回耗时；从模型 GPU 渲染开始到解码完成的帧处理耗时。

帧处理耗时包含模型 GPU 渲染/读回、输入打包、可能的单任务调度等待、NPU 运行与后处理；不包含相机曝光、相机内部管线、等待取最新纹理和屏幕显示。相机回报、GPU 取帧和检测 FPS 分开统计，不能用其中一项代替另一项。超过 1.5 秒没有新样本时帧率归零。

相机仅在检测页面可见时运行；退出页面或切到后台释放相机与 NPU，返回时重新建立。没有录屏、通知或录音权限。应用不保存、上传相机帧，不录制视频。图片检测和两种模型支持仍保留；NPU 失败仍直接报错。

## 验证与限制

- 21 项主机测试通过：原有 12 项和新增的 9 项（固定 120 模式选择、拒绝变动/不兼容 FPS、旋转与奇数 padding、RGBA 行倒序/通道/stride、截断缓冲拒绝、相机与检测 FPS 独立统计）。
- 主 APK、设备测试 APK 构建和 APK 签名验证通过；Android Lint 0 errors，保留配置/中文字符串等警告，完整日志在 .deps/camera-build-040.log。
- 没有连接手机，未完成 Camera2 120 FPS 会话、外部纹理方向、实景框位置、NPU 持续运行与性能的真机验证。主机测试和构建成功不代表已证明设备上达到 120 FPS。
- 两份 224 模型及其精度报告沿用 0.3.0，哈希保持一致。FP16 权重版并非已验证 NPU 内部 FP16 执行。

主要文件：Camera2Source.java、CameraGlView.java、CameraDetectionActivity.java、DetectionOverlay.java、NpuDetector.detectRgba、CameraPipelineTest.java。

实现复用 Android Camera2 constrained-high-speed API、SurfaceTexture 和 OpenGL ES 2.0；没有引入新的摄像头 SDK 或自定义 NPU 后端。依据 Android AOSP CameraDevice 与 CameraConstrainedHighSpeedCaptureSession 文档核对单 SurfaceTexture 高速输出约束：

- https://developer.android.com/reference/android/hardware/camera2/CameraConstrainedHighSpeedCaptureSession
- https://developer.android.com/reference/android/hardware/camera2/params/StreamConfigurationMap#getHighSpeedVideoFpsRangesFor(android.util.Size)

源码工程：C:/Users/Alienware/Desktop/fydnn/android-dimensity9500。构建方法见 MIGRATION.md。
