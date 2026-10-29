package com.fydnn.dimensity;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import com.google.ai.edge.litert.TensorBuffer;
import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.List;

/** Same JPEG, input tensors, compiled model and buffer objects, alternating original/new CPU paths. */
final class PipelineDeviceBenchmark {
    private static Object field(Object owner,String name)throws Exception {
        Field f=NpuDetector.class.getDeclaredField(name);f.setAccessible(true);return f.get(owner);
    }
    static String run(NpuDetector detector,byte[] jpeg,int[] outputShape,boolean floatingOutput,
                      float outputScale,int outputZero,boolean unsigned,YoloDecoder.Family family,
                      YoloDecoder.Layout layout,int classes,boolean normalized)throws Exception {
        float[] floats=(float[])field(detector,"inputFloats"),lookup=(float[])field(detector,"rgbLookup");
        byte[] bytes=(byte[])field(detector,"inputBytes"),quant=(byte[])field(detector,"quantLookup");
        TensorBuffer input=(TensorBuffer)field(detector,"inputBuffer"),output=(TensorBuffer)field(detector,"outputBuffer");
        PreparedNpuCall call=(PreparedNpuCall)field(detector,"preparedCall");
        int w=detector.width,h=detector.height;int[] pixels=new int[w*h];
        Bitmap resized=Bitmap.createBitmap(w,h,Bitmap.Config.ARGB_8888);Canvas canvas=new Canvas(resized);
        Paint paint=new Paint(Paint.FILTER_BITMAP_FLAG);Rect rect=new Rect();
        var f=BaselineYoloDecoder.Family.valueOf(family.name());var l=BaselineYoloDecoder.Layout.valueOf(layout.name());
        double[] baseline=new double[60],optimized=new double[60];List<BaselineYoloDecoder.Box> expected=null;
        try {
            for(int i=-10;i<60;i++)for(int j=0;j<2;j++) {
                boolean fast=((i+j)&1)!=0;long start=System.nanoTime();
                BitmapFactory.Options bounds=new BitmapFactory.Options();bounds.inJustDecodeBounds=true;
                BitmapFactory.decodeByteArray(jpeg,0,jpeg.length,bounds);
                BitmapFactory.Options options=new BitmapFactory.Options();options.inPreferredConfig=Bitmap.Config.ARGB_8888;
                Bitmap image=BitmapFactory.decodeByteArray(jpeg,0,jpeg.length,options);
                if(image==null)throw new AssertionError("JPEG decode failed");
                try {
                    if(fast) {
                        var result=detector.detect(image);
                        if(expected!=null) {
                            if(expected.size()!=result.boxes().size())throw new AssertionError("Pipeline detection count changed");
                            for(int n=0;n<expected.size();n++) {
                                var a=expected.get(n);var b=result.boxes().get(n);
                                if(a.cls()!=b.cls() || a.score()!=b.score() || a.left()!=b.left()
                                        || a.top()!=b.top() || a.right()!=b.right() || a.bottom()!=b.bottom())
                                    throw new AssertionError("Pipeline boxes changed");
                            }
                        }
                    } else {
                        float scale=Math.min((float)w/image.getWidth(),(float)h/image.getHeight());
                        int rw=Math.max(1,Math.round(image.getWidth()*scale)),rh=Math.max(1,Math.round(image.getHeight()*scale));
                        int px=(w-rw)/2,py=(h-rh)/2;canvas.drawColor(Color.rgb(114,114,114));rect.set(px,py,px+rw,py+rh);
                        canvas.drawBitmap(image,null,rect,paint);resized.getPixels(pixels,0,w,0,0,w,h);
                        int k=0;
                        for(int pixel:pixels)for(int shift=16;shift>=0;shift-=8) {
                            int value=(pixel>>shift)&255;
                            if(floats!=null)floats[k++]=lookup[value];else bytes[k++]=quant[value];
                        }
                        if(floats!=null)input.writeFloat(floats);else input.writeInt8(bytes);
                        call.run();var data=floatingOutput?BaselineYoloDecoder.Values.floats(output.readFloat()):
                                BaselineYoloDecoder.Values.quantized(output.readInt8(),outputScale,outputZero,unsigned);
                        expected=BaselineYoloDecoder.decode(data,outputShape,f,l,classes,normalized,w,h,
                                image.getWidth(),image.getHeight(),rw,rh,px,py,.25f,.45f);
                    }
                    if(i>=0)(fast?optimized:baseline)[i]=(System.nanoTime()-start)/1e6;
                } finally {image.recycle();}
            }
        } finally {resized.recycle();}
        return "local_pipeline_boxes_exact_match=true\nlocal_pipeline_scope=JPEG decode + letterbox/RGB + NPU/sync readback + decoder; excludes camera capture and display"
                +"\nlocal_pipeline_baseline_avg_p50_p95_ms="+stats(baseline)
                +"\nlocal_pipeline_optimized_avg_p50_p95_ms="+stats(optimized);
    }
    private static String stats(double[] data) {Arrays.sort(data);return Arrays.toString(new double[]{
            Arrays.stream(data).average().orElseThrow(),data[(data.length-1)/2],data[(int)Math.ceil(data.length*.95)-1]});}
}
