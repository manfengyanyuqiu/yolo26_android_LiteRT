package com.fydnn.dimensity;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageDecoder;
import android.graphics.Paint;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.OpenableColumns;
import android.widget.*;
import android.database.Cursor;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.tensorflow.lite.Interpreter;

public final class MainActivity extends Activity {
    private static final int PICK_MODEL=1, PICK_IMAGE=2, CAMERA_PERMISSION=3;
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private NpuDetector detector;
    private Bitmap image;
    private File modelFile;
    private String modelFailure;
    private Spinner family, layout;
    private EditText classes;
    private CheckBox normalized;
    private TextView status, modelLabel;
    private ImageView preview;
    private Button load, pickModel, pickImage, detect, benchmark, capture;
    private volatile boolean destroyed;

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        LinearLayout content = new LinearLayout(this); content.setOrientation(LinearLayout.VERTICAL);
        int padding=(int)(16*getResources().getDisplayMetrics().density);
        content.setPadding(padding,padding,padding,padding);
        content.setOnApplyWindowInsetsListener((v,insets)-> {
            android.graphics.Insets b=insets.getInsets(android.view.WindowInsets.Type.systemBars());
            v.setPadding(padding+b.left,padding+b.top,padding+b.right,padding+b.bottom); return insets;
        });
        ScrollView scroll=new ScrollView(this); scroll.addView(content); setContentView(scroll);
        TextView title=new TextView(this); title.setText("天玑 NPU 推理"); title.setTextSize(24); content.addView(title);
        modelLabel=new TextView(this); modelLabel.setText("先选择原始模型导出的 .tflite；.fymf 不受此运行时支持"); content.addView(modelLabel);
        family=spinner(content,new String[]{"YOLOv5（含 objectness）","YOLOv8（无 objectness）",
                "YOLO26（密集输出 + NMS）","YOLO26（端到端 xyxy/分数/类别）"});
        layout=spinner(content,new String[]{"输出 [1,N,C]","输出 [1,C,N]"});
        classes=new EditText(this); classes.setHint("类别数量（须与模型一致）"); classes.setText("80");
        classes.setInputType(android.text.InputType.TYPE_CLASS_NUMBER); content.addView(classes);
        normalized=new CheckBox(this); normalized.setText("模型使用归一化坐标（发布包自动识别；自定义模型请按导出设置核对）");
        normalized.setChecked(false); content.addView(normalized);
        pickModel=button(content,"1. 选择 TFLite 模型",()->pick(PICK_MODEL,"*/*"));
        load=button(content,"2. 编译到联发科 NPU",this::loadModel);
        pickImage=button(content,"3. 选择测试图片",()->pick(PICK_IMAGE,"image/*"));
        detect=button(content,"检测当前图片",this::detectImage);
        benchmark=button(content,"测速：5 次预热 + 50 次推理",this::benchmarkImage);
        capture=button(content,"摄像头检测：120 FPS 采集 + 实时画框",this::beginCapture);
        status=new TextView(this); status.setText(Build.MANUFACTURER+" / "+Build.MODEL+" / "+Build.SOC_MODEL
                +"\n等待模型。仅请求 NPU；自动 CPU 回退关闭。"); content.addView(status);
        preview=new ImageView(this); preview.setAdjustViewBounds(true); content.addView(preview);
    }
    private Spinner spinner(LinearLayout parent,String[] entries){
        Spinner s=new Spinner(this);s.setAdapter(new ArrayAdapter<>(this,android.R.layout.simple_spinner_dropdown_item,entries));
        parent.addView(s);return s;
    }
    private Button button(LinearLayout parent,String text,Runnable action){
        Button b=new Button(this);b.setText(text);b.setOnClickListener(v->action.run());parent.addView(b);return b;
    }
    private void pick(int request,String mime){
        Intent i=new Intent(Intent.ACTION_OPEN_DOCUMENT).setType(mime).addCategory(Intent.CATEGORY_OPENABLE);
        startActivityForResult(i,request);
    }
    private void beginCapture(){
        if(modelFile==null){status.setText("先选择 TFLite 模型");return;}
        if(checkSelfPermission(android.Manifest.permission.CAMERA)
                !=android.content.pm.PackageManager.PERMISSION_GRANTED){
            requestPermissions(new String[]{android.Manifest.permission.CAMERA},CAMERA_PERMISSION);
            return;
        }
        final int count;
        try{count=Integer.parseInt(classes.getText().toString());if(count<1)throw new NumberFormatException();}
        catch(NumberFormatException e){status.setText("类别数量需要是正整数");return;}
        Intent camera=new Intent(this,CameraDetectionActivity.class)
                .putExtra("model",modelFile.getAbsolutePath()).putExtra("family",family.getSelectedItemPosition())
                .putExtra("layout",layout.getSelectedItemPosition()).putExtra("classes",count)
                .putExtra("normalized",normalized.isChecked());
        busy(true,"准备摄像头检测…");
        worker.execute(()->{
            if(detector!=null){detector.close();detector=null;}
            runOnUiThread(()->{if(!destroyed){busy(false,"摄像头检测退出后可重新进入，或编译模型进行图片检测。");startActivity(camera);}});
        });
    }
    @Override public void onRequestPermissionsResult(int request,String[] permissions,int[] results){
        super.onRequestPermissionsResult(request,permissions,results);
        if(request==CAMERA_PERMISSION){
            if(results.length>0 && results[0]==android.content.pm.PackageManager.PERMISSION_GRANTED)beginCapture();
            else status.setText("摄像头检测需要相机权限，请在应用权限中开启。");
        }
    }
    @Override public void onActivityResult(int request,int result,Intent data){
        super.onActivityResult(request,result,data);
        if(result!=RESULT_OK || data==null)return;
        Uri uri=data.getData();if(uri==null)return;
        final int selectedFamily=family.getSelectedItemPosition();
        final int selectedClasses;
        try { selectedClasses=Integer.parseInt(classes.getText().toString());if(selectedClasses<1)throw new NumberFormatException(); }
        catch(NumberFormatException e){status.setText("类别数量需要是正整数");return;}
        busy(true,"读取文件…");
        worker.execute(()-> {
            try {
                if(request==PICK_MODEL){
                    String name="";
                    try(Cursor cursor=getContentResolver().query(uri,new String[]{OpenableColumns.DISPLAY_NAME},null,null,null)){
                        if(cursor!=null && cursor.moveToFirst())name=cursor.getString(0);
                    }
                    if(!name.toLowerCase(Locale.ROOT).endsWith(".tflite"))
                        throw new IllegalArgumentException("请选择 .tflite 模型；原始 .fymf 需从训练模型重新导出");
                    File file=new File(getFilesDir(),"detector.tflite");
                    modelFile=null;modelFailure=null;
                    if(detector!=null){detector.close();detector=null;}
                    try(InputStream in=getContentResolver().openInputStream(uri);FileOutputStream out=new FileOutputStream(file)){
                        if(in==null)throw new IllegalStateException("文件读取失败");
                        android.os.FileUtils.copy(in,out);
                    }
                    ModelImportOptions options;
                    try(Interpreter metadata=new Interpreter(file,new Interpreter.Options().setNumThreads(1))){
                        if(metadata.getOutputTensorCount()!=1)
                            throw new IllegalArgumentException("请选择单输出检测模型");
                        options=ModelImportOptions.resolve(name,metadata.getOutputTensor(0).shape(),selectedFamily,selectedClasses);
                    }
                    modelFile=file;modelFailure=null;String label=name;
                    runOnUiThread(()->{
                        modelLabel.setText(label);
                        family.setSelection(options.familyIndex());
                        layout.setSelection(options.layoutIndex());
                        normalized.setChecked(options.normalized());
                    });
                    finishTask("模型已读取，输出布局已按张量识别。请核对系列、类别数及归一化坐标后编译到 NPU。");
                }else if(request==PICK_IMAGE){
                    Bitmap decoded=ImageDecoder.decodeBitmap(ImageDecoder.createSource(getContentResolver(),uri),
                        (decoder,info,source)->decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE));
                    Bitmap old=image;image=decoded;
                    runOnUiThread(()->{preview.setImageBitmap(decoded);if(old!=null)old.recycle();});
                    finishTask("图片已加载："+decoded.getWidth()+"×"+decoded.getHeight());
                }
            }catch(Exception|LinkageError e){finishTask("读取失败："+e);}
        });
    }
    private void loadModel(){
        if(modelFile==null){status.setText("先选择 .tflite 模型");return;}
        final int count;
        try{count=Integer.parseInt(classes.getText().toString());if(count<1)throw new NumberFormatException();}
        catch(NumberFormatException e){status.setText("类别数量需要是正整数");return;}
        YoloDecoder.Family f=YoloDecoder.Family.values()[family.getSelectedItemPosition()];
        YoloDecoder.Layout l=YoloDecoder.Layout.values()[layout.getSelectedItemPosition()];
        boolean norm=normalized.isChecked();
        busy(true,"NPU 编译中，首次加载可能较慢…");
        worker.execute(()->{
            try{
                modelFailure=null;
                if(detector!=null){detector.close();detector=null;}
                detector=new NpuDetector(this,modelFile,f,l,count,norm);finishTask(detector.description);
            }catch(Exception|LinkageError e){
                modelFailure=(e instanceof IllegalArgumentException ? "模型参数校验失败：" : "NPU 编译或首次执行失败：")+e;
                finishTask(modelFailure+"\n详细原因已记录到 DIMENSITY-NPU 日志，未切换 CPU。");
            }
        });
    }
    private void detectImage(){
        busy(true,"检测中…");
        worker.execute(()->{
            try{
                requireReady();NpuDetector.Result r=detector.detect(image);
                Bitmap drawn=image.copy(Bitmap.Config.ARGB_8888,true);Canvas c=new Canvas(drawn);
                Paint p=new Paint(Paint.ANTI_ALIAS_FLAG);p.setColor(Color.GREEN);p.setStrokeWidth(3);p.setTextSize(24);
                for(YoloDecoder.Box box:r.boxes()){
                    p.setStyle(Paint.Style.STROKE);c.drawRect(box.left(),box.top(),box.right(),box.bottom(),p);
                    p.setStyle(Paint.Style.FILL);c.drawText(box.cls()+String.format(Locale.US," %.2f",box.score()),box.left(),Math.max(24,box.top()),p);
                }
                runOnUiThread(()->{if(!destroyed)preview.setImageBitmap(drawn);else drawn.recycle();});
                finishTask(String.format(Locale.US,"%d 个目标\n预处理 %.2f ms / run %.2f ms / 读回 %.2f ms / 后处理 %.2f ms\n端到端 %.2f ms",
                    r.boxes().size(),r.preprocessMs(),r.inferenceMs(),r.readbackMs(),r.postprocessMs(),
                    r.preprocessMs()+r.inferenceMs()+r.readbackMs()+r.postprocessMs()));
            }catch(Exception|LinkageError e){finishTask("检测失败："+e);}
        });
    }
    private void benchmarkImage(){
        busy(true,"预热并测速…");
        worker.execute(()->{try{requireReady();double[] t=detector.benchmark(image,5,50);
            double[] end=detector.benchmarkEndToEnd(image,5,50);
            finishTask(String.format(Locale.US,"run + 同步读回，50 次\n平均 %.2f ms / P50 %.2f ms / P95 %.2f ms\n后端串行吞吐 %.1f FPS\n图片端到端：平均 %.2f ms / P50 %.2f ms / P95 %.2f ms\n图片吞吐 %.1f FPS（不含采集和显示）",t[0],t[1],t[2],1000/t[0],end[0],end[1],end[2],1000/end[0]));
        }catch(Exception|LinkageError e){finishTask("测速失败："+e);}});
    }
    private void requireReady(){
        if(modelFile==null)throw new IllegalStateException("请先选择 TFLite 模型");
        if(detector==null)throw new IllegalStateException(modelFailure!=null ? modelFailure : "当前模型尚未就绪，请点击“编译到联发科 NPU”");
        if(image==null)throw new IllegalStateException("NPU 模型已就绪，请先选择测试图片");
    }
    private void busy(boolean busy,String message){
        for(Button b:new Button[]{load,pickModel,pickImage,detect,benchmark,capture})b.setEnabled(!busy);
        family.setEnabled(!busy);layout.setEnabled(!busy);classes.setEnabled(!busy);normalized.setEnabled(!busy);status.setText(message);
    }
    private void finishTask(String message){runOnUiThread(()->{if(!destroyed)busy(false,message);});}
    @Override public void onDestroy(){
        destroyed=true;worker.execute(()->{if(detector!=null)detector.close();});worker.shutdown();super.onDestroy();
    }
}
