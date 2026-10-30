package com.fydnn.dimensity;

import android.app.Instrumentation;
import android.app.Activity;
import android.content.ContentValues;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.widget.*;
import java.io.*;
import java.lang.reflect.Field;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;

/** Exercise the actual document-result handler and visible compile/detect buttons. */
final class ImportDeviceCheck {
    private static Object field(MainActivity activity,String name)throws Exception {
        Field f=MainActivity.class.getDeclaredField(name);f.setAccessible(true);return f.get(activity);
    }
    private static Uri document(Instrumentation test,File source,String name,String mime)throws Exception {
        ContentValues values=new ContentValues();
        values.put(MediaStore.MediaColumns.DISPLAY_NAME,name);
        values.put(MediaStore.MediaColumns.MIME_TYPE,mime);
        values.put(MediaStore.MediaColumns.RELATIVE_PATH,"Download/YoloNpuImportTests");
        var resolver=test.getTargetContext().getContentResolver();
        Uri uri=resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI,values);
        if(uri==null)throw new AssertionError("Test document creation failed");
        try(InputStream in=new FileInputStream(source);OutputStream out=resolver.openOutputStream(uri)) {
            if(out==null)throw new AssertionError("Test document write failed");
            in.transferTo(out);
        }
        return uri;
    }
    private static void idle(Instrumentation test,MainActivity activity)throws Exception {
        ((ExecutorService)field(activity,"worker")).submit(()->{}).get(45,TimeUnit.SECONDS);
        test.waitForIdleSync();
    }
    static void run(Instrumentation test,Bundle args,PrintWriter writer)throws Exception {
        MainActivity activity=(MainActivity)test.startActivitySync(new Intent(test.getTargetContext(),MainActivity.class)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK|Intent.FLAG_ACTIVITY_CLEAR_TASK));
        Uri model=null,image=null;
        try {
            String filename=args.getString("filename");
            File source=new File(Boolean.parseBoolean(args.getString("externalModel","false"))
                    ? test.getTargetContext().getExternalFilesDir("diagnostics") : test.getTargetContext().getFilesDir(),args.getString("model"));
            String expectedHash=args.getString("modelSha256");
            if(expectedHash!=null) {
                byte[] hash=java.security.MessageDigest.getInstance("SHA-256").digest(java.nio.file.Files.readAllBytes(source.toPath()));
                StringBuilder hex=new StringBuilder();for(byte b:hash)hex.append(String.format(java.util.Locale.ROOT,"%02x",b&255));
                if(!expectedHash.contentEquals(hex))throw new AssertionError("Staged model SHA256 mismatch");
            }
            model=document(test,source,filename,"application/octet-stream");
            image=document(test,new File(test.getTargetContext().getFilesDir(),args.getString("image")),"import-check.jpg","image/jpeg");
            final Uri selectedModel=model,selectedImage=image;
            test.runOnMainSync(()->activity.onActivityResult(1,Activity.RESULT_OK,new Intent().setData(selectedModel)));
            idle(test,activity);
            if(field(activity,"modelFile")==null)throw new AssertionError("Import failed: "+((TextView)field(activity,"status")).getText());
            int family=((Spinner)field(activity,"family")).getSelectedItemPosition();
            int layout=((Spinner)field(activity,"layout")).getSelectedItemPosition();
            boolean normalized=((CheckBox)field(activity,"normalized")).isChecked();
            writer.println("imported="+filename+" family="+family+" layout="+layout+" normalized="+normalized);writer.flush();
            if(family!=Integer.parseInt(args.getString("expectedFamily"))
                    ||layout!=Integer.parseInt(args.getString("expectedLayout"))||!normalized)
                throw new AssertionError("Model picker selected incorrect options");
            Button compile=(Button)field(activity,"load");
            test.runOnMainSync(compile::performClick);idle(test,activity);
            NpuDetector detector=(NpuDetector)field(activity,"detector");
            if(detector==null)throw new AssertionError("Compile button failed: "+((TextView)field(activity,"status")).getText());
            writer.println(detector.description);writer.flush();
            test.runOnMainSync(()->activity.onActivityResult(2,Activity.RESULT_OK,new Intent().setData(selectedImage)));
            idle(test,activity);
            Button detect=(Button)field(activity,"detect");
            test.runOnMainSync(detect::performClick);idle(test,activity);
            String status=((TextView)field(activity,"status")).getText().toString();
            if(!status.contains("个目标"))throw new AssertionError("Detect button failed: "+status);
            writer.println(status);writer.flush();
            String tensorTag=args.getString("dumpTensors");
            if(tensorTag!=null) {
                if(!tensorTag.matches("[A-Za-z0-9_-]+"))throw new IllegalArgumentException("Invalid tensor tag");
                File directory=test.getTargetContext().getExternalFilesDir("diagnostics");
                try(org.tensorflow.lite.Interpreter metadata=new org.tensorflow.lite.Interpreter(source)) {
                    for(String name:new String[]{"inputBuffer","outputBuffer"}) {
                        Field f=NpuDetector.class.getDeclaredField(name);f.setAccessible(true);
                        com.google.ai.edge.litert.TensorBuffer buffer=(com.google.ai.edge.litert.TensorBuffer)f.get(detector);
                        var spec=name.equals("inputBuffer")?metadata.getInputTensor(0):metadata.getOutputTensor(0);
                        byte[] bytes;
                        if(spec.dataType()==org.tensorflow.lite.DataType.FLOAT32) {
                            float[] values=buffer.readFloat();
                            java.nio.ByteBuffer raw=java.nio.ByteBuffer.allocate(values.length*4).order(java.nio.ByteOrder.LITTLE_ENDIAN);
                            raw.asFloatBuffer().put(values);bytes=raw.array();
                        }else bytes=buffer.readInt8();
                        File destination=new File(directory,tensorTag+"_"+name+".bin");
                        try(FileOutputStream stream=new FileOutputStream(destination)){stream.write(bytes);}
                        writer.println("tensor="+destination.getAbsolutePath());writer.flush();
                    }
                }
            }
            if(Boolean.parseBoolean(args.getString("saveGallery","false"))) {
                Bitmap[] snapshot=new Bitmap[1];
                test.runOnMainSync(()->{
                    try {
                        ImageView preview=(ImageView)field(activity,"preview");
                        snapshot[0]=((BitmapDrawable)preview.getDrawable()).getBitmap().copy(Bitmap.Config.ARGB_8888,false);
                    }catch(Exception e){throw new IllegalStateException("Result preview snapshot failed",e);}
                });
                try {
                    Uri saved=GalleryResultImage.save(test.getTargetContext(),snapshot[0],filename,status,
                            detector.width,detector.height,args.getString("runId"));
                    writer.println("gallery_uri="+saved);writer.flush();
                }finally{snapshot[0].recycle();}
            }
            if(Boolean.parseBoolean(args.getString("benchmarkUi","false"))) {
                Button benchmark=(Button)field(activity,"benchmark");
                test.runOnMainSync(benchmark::performClick);idle(test,activity);
                String timing=((TextView)field(activity,"status")).getText().toString();
                if(!timing.contains("50 次"))throw new AssertionError("Benchmark failed: "+timing);
                writer.println("benchmark_ui="+timing);writer.flush();
            }
        } finally {
            var resolver=test.getTargetContext().getContentResolver();
            if(model!=null)resolver.delete(model,null,null);
            if(image!=null)resolver.delete(image,null,null);
            test.runOnMainSync(activity::finish);
        }
    }
}
