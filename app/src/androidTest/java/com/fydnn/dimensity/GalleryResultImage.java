package com.fydnn.dimensity;

import android.content.ContentValues;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.net.Uri;
import android.provider.MediaStore;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

/** Save the real UI detection image with a separate model/timing caption. */
final class GalleryResultImage {
    static Uri save(Context context,Bitmap detection,String modelName,String status,
                    int width,int height,String runId)throws Exception {
        if(!runId.matches("[A-Za-z0-9_-]+"))throw new IllegalArgumentException("Invalid run ID");
        Paint text=new Paint(Paint.ANTI_ALIAS_FLAG);
        text.setColor(Color.WHITE);text.setTextSize(20);
        List<String> lines=new ArrayList<>();
        lines.add("Model: "+modelName);
        lines.add("Input: "+width+" x "+height+" | MediaTek Neuron NPU");
        for(String line:status.split("\n"))lines.add(line);
        List<String> wrapped=new ArrayList<>();
        for(String line:lines) {
            while(!line.isEmpty()) {
                int count=text.breakText(line,true,detection.getWidth()-24,null);
                if(count==0)throw new IllegalArgumentException("Result image too narrow for caption");
                wrapped.add(line.substring(0,count));line=line.substring(count);
            }
        }
        int header=24+wrapped.size()*28;
        Bitmap labeled=Bitmap.createBitmap(detection.getWidth(),detection.getHeight()+header,Bitmap.Config.ARGB_8888);
        try {
            Canvas canvas=new Canvas(labeled);canvas.drawColor(Color.rgb(24,24,24));
            for(int i=0;i<wrapped.size();i++)canvas.drawText(wrapped.get(i),12,28+i*28,text);
            canvas.drawBitmap(detection,0,header,null);
            ContentValues values=new ContentValues();
            String filename=runId+"_"+modelName.replaceFirst("(?i)\\.tflite$","")+".png";
            values.put(MediaStore.Images.Media.DISPLAY_NAME,filename);
            values.put(MediaStore.Images.Media.MIME_TYPE,"image/png");
            values.put(MediaStore.Images.Media.RELATIVE_PATH,"Pictures/YOLO_NPU_Tests");
            values.put(MediaStore.Images.Media.IS_PENDING,1);
            var resolver=context.getContentResolver();
            Uri uri=resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI,values);
            if(uri==null)throw new IllegalStateException("Gallery insertion failed");
            try {
                try(OutputStream out=resolver.openOutputStream(uri)) {
                    if(out==null||!labeled.compress(Bitmap.CompressFormat.PNG,100,out))
                        throw new IllegalStateException("Gallery PNG write failed");
                }
                values.clear();values.put(MediaStore.Images.Media.IS_PENDING,0);
                if(resolver.update(uri,values,null,null)!=1)throw new IllegalStateException("Gallery publish failed");
                return uri;
            }catch(Exception e){resolver.delete(uri,null,null);throw e;}
        }finally{labeled.recycle();}
    }
}
