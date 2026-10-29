package com.fydnn.dimensity;

import android.app.Activity;
import android.graphics.Color;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.Gravity;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;
import java.io.File;
import java.nio.ByteBuffer;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

/** Visible camera use only; backgrounding releases both camera and the NPU on their owning threads. */
public final class CameraDetectionActivity extends Activity {
    private final ExecutorService inference=Executors.newSingleThreadExecutor();
    private final Handler ui=new Handler(Looper.getMainLooper());
    private volatile boolean active;
    private volatile int generation;
    private NpuDetector detector; // Inference worker owns create/run/close.
    private CameraGlView cameraView;
    private DetectionOverlay overlay;
    private FrameLayout previewContainer;
    private TextView status;
    private volatile CameraModeSelector.Mode mode;
    private FrameRateMeter sensorRate,previewRate,detectionRate;
    private volatile double readbackMs,backendMs,processingMs;
    private volatile int boxCount;
    private boolean running;

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        LinearLayout root=new LinearLayout(this); root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(Color.BLACK);
        int padding=(int)(12*getResources().getDisplayMetrics().density);
        root.setOnApplyWindowInsetsListener((v,insets) -> {
            var b=insets.getInsets(android.view.WindowInsets.Type.systemBars());
            v.setPadding(padding+b.left,b.top,padding+b.right,b.bottom); return insets;
        });
        status=new TextView(this); status.setTextColor(Color.WHITE); status.setTextSize(15);
        status.setMinLines(5); root.addView(status);
        FrameLayout stage=new FrameLayout(this);
        root.addView(stage,new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,0,1));
        previewContainer=new FrameLayout(this);
        stage.addView(previewContainer,new FrameLayout.LayoutParams(1,1,Gravity.CENTER));
        stage.addOnLayoutChangeListener((v,l,t,r,b,ol,ot,or,ob) -> resizePreview(stage));
        Button stop=new Button(this); stop.setText("停止摄像头检测并返回"); stop.setOnClickListener(v -> finish());
        root.addView(stop); setContentView(root);
    }

    private int inputWidth=224,inputHeight=224;
    private void resizePreview(FrameLayout stage) {
        int availableWidth=stage.getWidth(), availableHeight=stage.getHeight();
        if (availableWidth<=0 || availableHeight<=0) return;
        float ratio=Math.min((float)availableWidth/inputWidth,(float)availableHeight/inputHeight);
        FrameLayout.LayoutParams params=(FrameLayout.LayoutParams)previewContainer.getLayoutParams();
        int width=Math.max(1,Math.round(inputWidth*ratio)), height=Math.max(1,Math.round(inputHeight*ratio));
        if (params.width!=width || params.height!=height) {
            params.width=width; params.height=height; previewContainer.setLayoutParams(params);
        }
    }

    @Override protected void onResume() {
        super.onResume(); active=true; running=false; int token=++generation;
        sensorRate=new FrameRateMeter(); previewRate=new FrameRateMeter(); detectionRate=new FrameRateMeter();
        mode=null; status.setText("正在编译 NPU 模型…\n请求后置摄像头固定 120 FPS");
        inference.execute(() -> {
            android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_DISPLAY);
            try {
                if (!isCurrent(token)) return;
                String path=getIntent().getStringExtra("model");
                if (path==null) throw new IllegalArgumentException("未选择模型");
                NpuDetector loaded=new NpuDetector(this,new File(path),
                        YoloDecoder.Family.values()[getIntent().getIntExtra("family",0)],
                        YoloDecoder.Layout.values()[getIntent().getIntExtra("layout",0)],
                        getIntent().getIntExtra("classes",80),getIntent().getBooleanExtra("normalized",false));
                if (!isCurrent(token)) { loaded.close(); return; }
                detector=loaded;
                int width=loaded.width,height=loaded.height;
                ui.post(() -> { if (isCurrent(token)) startPreview(token,width,height); });
            } catch (Exception|LinkageError e) { reportError(token,"NPU 模型加载失败："+e); }
        });
    }

    private void startPreview(int token,int width,int height) {
        inputWidth=width; inputHeight=height;
        resizePreview((FrameLayout)previewContainer.getParent());
        overlay=new DetectionOverlay(this);
        DetectionOverlay targetOverlay=overlay;
        AtomicBoolean frameBusy=new AtomicBoolean();
        int displayRotation=getDisplay().getRotation()*90;
        cameraView=new CameraGlView(this,width,height,displayRotation,new CameraGlView.Listener() {
            @Override public void onMode(CameraModeSelector.Mode selected) { if (isCurrent(token)) mode=selected; }
            @Override public void onCapture(long time) { if (isCurrent(token)) sensorRate.add(time); }
            @Override public void onPreviewFrame(long time) { if (isCurrent(token)) previewRate.add(time); }
            @Override public void onError(String message) { reportError(token,message); }
            @Override public boolean acquireFrame() { return isCurrent(token) && frameBusy.compareAndSet(false,true); }
            @Override public void releaseFrame() { frameBusy.set(false); }
            @Override public void onFrame(ByteBuffer rgba,long start,double readMs) {
                inference.execute(() -> {
                    try {
                        if (!isCurrent(token) || detector==null) return;
                        NpuDetector.Result result=detector.detectRgba(rgba,width*4,true);
                        if (!isCurrent(token)) return;
                        detectionRate.add(System.nanoTime());
                        readbackMs=readMs; backendMs=result.inferenceMs()+result.readbackMs();
                        processingMs=(System.nanoTime()-start)/1e6; boxCount=result.boxes().size();
                        targetOverlay.update(result.boxes(),width,height);
                    } catch (Exception|LinkageError e) { reportError(token,"摄像头推理失败："+e); }
                    finally { frameBusy.set(false); }
                });
            }
        });
        previewContainer.addView(cameraView,new FrameLayout.LayoutParams(-1,-1));
        previewContainer.addView(overlay,new FrameLayout.LayoutParams(-1,-1));
        running=true; cameraView.onResume(); ui.post(updateStats);
    }

    private final Runnable updateStats=new Runnable() {
        @Override public void run() {
            if (!active || !running) return;
            CameraModeSelector.Mode current=mode;
            String camera=current==null?"正在配置 120 FPS 相机…":String.format(Locale.US,
                    "后置相机 %s / %d×%d / 请求 120 FPS / %s",current.cameraId(),current.width(),current.height(),
                    current.highSpeed()?"高速会话":"普通会话");
            status.setText(String.format(Locale.US,
                    "%s\n相机回报 %.1f / GPU取帧 %.1f / 检测 %.1f FPS\n%d 个目标 / GPU读回 %.2f ms\nNPU运行+输出读回 %.2f ms / 帧处理 %.2f ms\n模型 %d×%d · 仅NPU · 推理忙时取最新帧",
                    camera,sensorRate.fps(),previewRate.fps(),detectionRate.fps(),boxCount,
                    readbackMs,backendMs,processingMs,inputWidth,inputHeight));
            if (overlay!=null) overlay.invalidate(); // Expire boxes when no new result arrives.
            ui.postDelayed(this,500);
        }
    };

    private boolean isCurrent(int token) { return active && generation==token; }
    private void reportError(int token,String message) {
        ui.post(() -> {
            if (!isCurrent(token)) return;
            ++generation; stopPreview(); status.setText(message+"\n未降低采集帧率，未切换 CPU。返回后可重新进入。");
            inference.execute(this::closeDetector);
        });
    }
    private void stopPreview() {
        running=false; ui.removeCallbacks(updateStats);
        if (cameraView!=null) {
            cameraView.stopCamera(); cameraView.releaseGl(); cameraView.onPause(); cameraView=null;
        }
        if (overlay!=null) { overlay.clear(); overlay=null; }
        previewContainer.removeAllViews();
    }
    private void closeDetector() { if (detector!=null) { detector.close(); detector=null; } }
    @Override protected void onPause() {
        active=false; ++generation; stopPreview(); inference.execute(this::closeDetector);
        super.onPause();
    }
    @Override protected void onDestroy() { inference.shutdown(); super.onDestroy(); }
}
