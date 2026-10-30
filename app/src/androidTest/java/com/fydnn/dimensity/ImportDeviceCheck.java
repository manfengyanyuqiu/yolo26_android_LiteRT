package com.fydnn.dimensity;

import android.app.Instrumentation;
import android.app.Activity;
import android.content.ContentValues;
import android.content.Intent;
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
            model=document(test,new File(test.getTargetContext().getFilesDir(),args.getString("model")),filename,"application/octet-stream");
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
        } finally {
            var resolver=test.getTargetContext().getContentResolver();
            if(model!=null)resolver.delete(model,null,null);
            if(image!=null)resolver.delete(image,null,null);
            test.runOnMainSync(activity::finish);
        }
    }
}
