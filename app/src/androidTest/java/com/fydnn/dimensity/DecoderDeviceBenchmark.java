package com.fydnn.dimensity;
import java.util.Arrays;
import java.util.List;

/** Alternating A/B on the same real synchronized NPU output, no model rerun in decoder timing. */
final class DecoderDeviceBenchmark {
    static String run(float[] floats,byte[] bytes,float scale,int zero,boolean unsigned,int[] shape,
                      YoloDecoder.Family family,YoloDecoder.Layout layout,int classes,boolean normalized,
                      int width,int height,int originalWidth,int originalHeight) {
        var fast=new YoloDecoder.Workspace();
        var data=floats!=null?YoloDecoder.Values.floats(floats):YoloDecoder.Values.quantized(bytes,scale,zero,unsigned);
        var baseline=floats!=null?BaselineYoloDecoder.Values.floats(floats):
                BaselineYoloDecoder.Values.quantized(bytes,scale,zero,unsigned);
        float ratio=Math.min((float)width/originalWidth,(float)height/originalHeight);
        int rw=Math.max(1,Math.round(originalWidth*ratio)),rh=Math.max(1,Math.round(originalHeight*ratio));
        int px=(width-rw)/2,py=(height-rh)/2;
        var f=BaselineYoloDecoder.Family.valueOf(family.name());var l=BaselineYoloDecoder.Layout.valueOf(layout.name());
        var expected=BaselineYoloDecoder.decode(baseline,shape,f,l,classes,normalized,width,height,
                originalWidth,originalHeight,rw,rh,px,py,.25f,.45f);
        var actual=fast.decode(data,shape,family,layout,classes,normalized,width,height,
                originalWidth,originalHeight,rw,rh,px,py,.25f,.45f);
        if(expected.size()!=actual.size())throw new AssertionError("Decoder count changed");
        for(int i=0;i<actual.size();i++) {
            var a=expected.get(i);var b=actual.get(i);
            if(a.cls()!=b.cls() || Float.floatToIntBits(a.score())!=Float.floatToIntBits(b.score())
                    || a.left()!=b.left() || a.top()!=b.top() || a.right()!=b.right() || a.bottom()!=b.bottom())
                throw new AssertionError("Decoder output changed at "+i);
        }
        for(int i=0;i<10;i++) {
            BaselineYoloDecoder.decode(baseline,shape,f,l,classes,normalized,width,height,originalWidth,originalHeight,rw,rh,px,py,.25f,.45f);
            fast.decode(data,shape,family,layout,classes,normalized,width,height,originalWidth,originalHeight,rw,rh,px,py,.25f,.45f);
        }
        double[] oldTimes=new double[60],newTimes=new double[60];
        for(int i=0;i<60;i++)for(int j=0;j<2;j++) {
            boolean optimized=((i+j)&1)!=0;long start=System.nanoTime();
            List<?> result=optimized?fast.decode(data,shape,family,layout,classes,normalized,width,height,
                    originalWidth,originalHeight,rw,rh,px,py,.25f,.45f):
                    BaselineYoloDecoder.decode(baseline,shape,f,l,classes,normalized,width,height,
                    originalWidth,originalHeight,rw,rh,px,py,.25f,.45f);
            (optimized?newTimes:oldTimes)[i]=(System.nanoTime()-start)/1e6;
            if(result.size()!=expected.size())throw new AssertionError("Unstable decoder output");
        }
        return "decoder_exact_match=true\ndecoder_baseline_avg_p50_p95_ms="+stats(oldTimes)
                +"\ndecoder_optimized_avg_p50_p95_ms="+stats(newTimes)
                +"\ndecoder_candidates="+fast.candidates()+" nms_comparisons="+fast.comparisons()
                +" scan_last_ms="+fast.scanMs()+" nms_last_ms="+fast.nmsMs();
    }
    private static String stats(double[] data) {
        Arrays.sort(data);return Arrays.toString(new double[]{Arrays.stream(data).average().orElseThrow(),
                data[(data.length-1)/2],data[(int)Math.ceil(data.length*.95)-1]});
    }
}
