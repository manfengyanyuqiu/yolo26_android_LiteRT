package com.fydnn.dimensity;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.*;
import android.hardware.camera2.params.OutputConfiguration;
import android.hardware.camera2.params.SessionConfiguration;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Range;
import android.util.Size;
import android.view.Surface;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

/** Android Camera2 constrained-high-speed API, single SurfaceTexture output, fixed 120 FPS. */
final class Camera2Source implements AutoCloseable {
    interface Listener {
        void onMode(CameraModeSelector.Mode mode);
        void onCapture(long sensorTimestampNs);
        void onError(String message);
    }
    private final Context context;
    private final Listener listener;
    private final HandlerThread thread = new HandlerThread("fydnn-camera2");
    private final Handler handler;
    private CameraDevice device;
    private CameraCaptureSession session;
    private Surface output;
    private boolean opening;
    private volatile boolean closed;
    private final AtomicBoolean closeRequested=new AtomicBoolean();

    Camera2Source(Context context, Listener listener) {
        this.context=context; this.listener=listener;
        thread.start(); handler=new Handler(thread.getLooper());
    }

    void start(SurfaceTexture texture) {
        handler.post(() -> {
            if (closed) { release(); return; }
            try {
                if (context.checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED)
                    throw new SecurityException("请先授予相机权限");
                CameraManager manager=context.getSystemService(CameraManager.class);
                CameraModeSelector.Mode mode=findMode(manager);
                texture.setDefaultBufferSize(mode.width(),mode.height());
                output=new Surface(texture);
                listener.onMode(mode);
                opening=true;
                manager.openCamera(mode.cameraId(),new CameraDevice.StateCallback() {
                    @Override public void onOpened(CameraDevice camera) {
                        opening=false; device=camera;
                        if (closed) { camera.close(); return; }
                        configure(camera,mode);
                    }
                    @Override public void onDisconnected(CameraDevice camera) {
                        if (!closed) listener.onError("相机连接中断");
                        closed=true; camera.close();
                    }
                    @Override public void onError(CameraDevice camera,int error) {
                        if (!closed) listener.onError("Camera2 打开失败，错误码 " + error);
                        closed=true; camera.close();
                    }
                    @Override public void onClosed(CameraDevice camera) { release(); }
                },handler);
            } catch (Exception e) {
                opening=false; fail("120 FPS 相机启动失败："+e);
            }
        });
    }

    private static CameraModeSelector.Mode findMode(CameraManager manager) throws CameraAccessException {
        List<CameraModeSelector.Mode> modes=new ArrayList<>();
        for (String id:manager.getCameraIdList()) {
            CameraCharacteristics c=manager.getCameraCharacteristics(id);
            Integer facing=c.get(CameraCharacteristics.LENS_FACING);
            if (facing == null || facing != CameraCharacteristics.LENS_FACING_BACK) continue;
            StreamConfigurationMap map=c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            Integer rotation=c.get(CameraCharacteristics.SENSOR_ORIENTATION);
            if (map == null || rotation == null) continue;
            int[] caps=c.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES);
            boolean high=caps != null && Arrays.stream(caps).anyMatch(
                    v -> v==CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_CONSTRAINED_HIGH_SPEED_VIDEO);
            if (high) for (Size size:map.getHighSpeedVideoSizes())
                for (Range<Integer> range:map.getHighSpeedVideoFpsRangesFor(size))
                    modes.add(new CameraModeSelector.Mode(id,size.getWidth(),size.getHeight(),rotation,true,
                            range.getLower(),range.getUpper(),0));
            Range<Integer>[] ranges=c.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES);
            Size[] sizes=map.getOutputSizes(SurfaceTexture.class);
            if (ranges != null && sizes != null) for (Range<Integer> range:ranges)
                for (Size size:sizes) {
                    long duration=map.getOutputMinFrameDuration(SurfaceTexture.class,size);
                    modes.add(new CameraModeSelector.Mode(id,size.getWidth(),size.getHeight(),rotation,false,
                            range.getLower(),range.getUpper(),duration));
                }
        }
        return CameraModeSelector.select(modes);
    }

    private void configure(CameraDevice camera,CameraModeSelector.Mode mode) {
        try {
            SessionConfiguration config=new SessionConfiguration(
                    mode.highSpeed()?SessionConfiguration.SESSION_HIGH_SPEED:SessionConfiguration.SESSION_REGULAR,
                    List.of(new OutputConfiguration(output)),handler::post,new CameraCaptureSession.StateCallback() {
                @Override public void onConfigured(CameraCaptureSession configured) {
                    if (closed) { configured.close(); return; }
                    session=configured;
                    try {
                        CaptureRequest.Builder builder=camera.createCaptureRequest(CameraDevice.TEMPLATE_RECORD);
                        builder.addTarget(output);
                        builder.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE,new Range<>(120,120));
                        builder.set(CaptureRequest.CONTROL_MODE,CaptureRequest.CONTROL_MODE_AUTO);
                        builder.set(CaptureRequest.CONTROL_AF_MODE,CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO);
                        if (mode.highSpeed()) {
                            CameraConstrainedHighSpeedCaptureSession high=(CameraConstrainedHighSpeedCaptureSession)configured;
                            high.setRepeatingBurst(high.createHighSpeedRequestList(builder.build()),captureCallback,handler);
                        } else configured.setRepeatingRequest(builder.build(),captureCallback,handler);
                    } catch (Exception e) { fail("固定 120 FPS 请求失败："+e); }
                }
                @Override public void onConfigureFailed(CameraCaptureSession failed) {
                    failed.close(); fail("Camera2 拒绝 120 FPS 会话配置（未降低帧率）");
                }
            });
            camera.createCaptureSession(config);
        } catch (Exception e) { fail("120 FPS 会话创建失败："+e); }
    }

    private final CameraCaptureSession.CaptureCallback captureCallback=new CameraCaptureSession.CaptureCallback() {
        @Override public void onCaptureCompleted(CameraCaptureSession session,CaptureRequest request,TotalCaptureResult result) {
            Long time=result.get(CaptureResult.SENSOR_TIMESTAMP);
            if (!closed && time != null) listener.onCapture(time);
        }
        @Override public void onCaptureFailed(CameraCaptureSession session,CaptureRequest request,CaptureFailure failure) {
            if (!closed) fail("Camera2 采集失败，reason="+failure.getReason());
        }
    };

    private void fail(String message) {
        if (!closed) listener.onError(message);
        close();
    }
    @Override public void close() {
        if (!closeRequested.compareAndSet(false,true)) return;
        closed=true;
        handler.post(() -> {
            if (session != null) { session.close(); session=null; }
            if (device != null) { device.close(); device=null; }
            else if (!opening) release();
        });
    }
    private void release() {
        if (output != null) { output.release(); output=null; }
        thread.quitSafely();
    }
}
