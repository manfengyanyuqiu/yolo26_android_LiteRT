package com.fydnn.dimensity;
import org.junit.Test;
import java.util.Random;
import java.util.List;
import static org.junit.Assert.*;

public class DecoderParityTest {
    private static void compare(YoloDecoder.Workspace workspace,float[] rows,int count,int classes,
                                YoloDecoder.Family family,YoloDecoder.Layout layout,boolean quantized,boolean unsigned) {
        int channels=classes+(family==YoloDecoder.Family.YOLOV5?5:4);
        float[] data=new float[rows.length];
        for(int n=0;n<count;n++)for(int c=0;c<channels;c++)
            data[layout==YoloDecoder.Layout.ROWS?n*channels+c:c*count+n]=rows[n*channels+c];
        int[] shape=layout==YoloDecoder.Layout.ROWS?new int[]{1,1,count,channels}:new int[]{1,1,channels,count};
        YoloDecoder.Values actual=YoloDecoder.Values.floats(data);
        BaselineYoloDecoder.Values reference=BaselineYoloDecoder.Values.floats(data);
        if(quantized) {
            byte[] raw=new byte[data.length];int zero=unsigned?0:-128;
            for(int i=0;i<data.length;i++)raw[i]=(byte)(Math.max(0,Math.min(255,Math.round(data[i]/.005f)))+zero);
            actual=YoloDecoder.Values.quantized(raw,.005f,zero,unsigned);
            reference=BaselineYoloDecoder.Values.quantized(raw,.005f,zero,unsigned);
        }
        List<BaselineYoloDecoder.Box> expected=BaselineYoloDecoder.decode(reference,shape,
                BaselineYoloDecoder.Family.valueOf(family.name()),BaselineYoloDecoder.Layout.valueOf(layout.name()),
                classes,true,320,320,1280,721,320,180,0,70,.25f,.45f);
        var result=workspace.decode(actual,shape,family,layout,classes,true,320,320,
                1280,721,320,180,0,70,.25f,.45f);
        assertEquals(expected.size(),result.size());
        for(int i=0;i<result.size();i++) {
            var a=expected.get(i);var b=result.get(i);
            assertEquals(a.cls(),b.cls());assertEquals(a.score(),b.score(),0);
            assertEquals(a.left(),b.left(),0);assertEquals(a.top(),b.top(),0);
            assertEquals(a.right(),b.right(),0);assertEquals(a.bottom(),b.bottom(),0);
        }
    }
    @Test public void randomizedLayoutsQuantizationAndWorkspaceReuseAreExact() {
        Random random=new Random(6204);YoloDecoder.Workspace workspace=new YoloDecoder.Workspace();
        for(var family:new YoloDecoder.Family[]{YoloDecoder.Family.YOLOV5,YoloDecoder.Family.YOLOV8,YoloDecoder.Family.YOLO26})
            for(int classes:new int[]{1,2,80})for(int count:new int[]{1,37,2100}) {
                int channels=classes+(family==YoloDecoder.Family.YOLOV5?5:4);
                float[] rows=new float[count*channels];
                for(int n=0;n<count;n++)for(int c=0;c<channels;c++)
                    rows[n*channels+c]=c<4?.1f+random.nextFloat()*.8f:Math.round(random.nextFloat()*100)/100f;
                for(var layout:YoloDecoder.Layout.values()) {
                    compare(workspace,rows,count,classes,family,layout,false,false);
                    compare(workspace,rows,count,classes,family,layout,true,false);
                    compare(workspace,rows,count,classes,family,layout,true,true);
                }
            }
    }
    @Test public void tiesThresholdAndThreeHundredLimitPreserveOrder() {
        int count=600,classes=2,channels=6;float[] rows=new float[count*channels];
        for(int n=0;n<count;n++) {
            int k=n*channels;rows[k]=(n%30+.5f)/30;rows[k+1]=(n/30+.5f)/20;
            rows[k+2]=.01f;rows[k+3]=.01f;rows[k+4]=.25f;rows[k+5]=.25f;
        }
        for(var layout:YoloDecoder.Layout.values())
            compare(new YoloDecoder.Workspace(),rows,count,classes,YoloDecoder.Family.YOLOV8,layout,false,false);
    }
    @Test public void lateInvalidClassProbabilityStillFails() {
        float[] data={.5f,.5f,.2f,.2f,.9f,Float.NaN};
        assertThrows(IllegalArgumentException.class,()->new YoloDecoder.Workspace().decode(YoloDecoder.Values.floats(data),
                new int[]{1,6,1},YoloDecoder.Family.YOLOV8,YoloDecoder.Layout.CHANNELS,2,true,
                320,320,320,320,320,320,0,0,.25f,.45f));
    }
    @Test public void infinityAndOutOfRangeProbabilitiesStillFailInEveryStoragePath() {
        for(var layout:YoloDecoder.Layout.values())for(float bad:new float[]{Float.NaN,Float.POSITIVE_INFINITY,Float.NEGATIVE_INFINITY,-.02f,1.02f}) {
            float[] data={.5f,.5f,.2f,.2f,.9f,bad};
            int[] shape=layout==YoloDecoder.Layout.ROWS?new int[]{1,1,6}:new int[]{1,6,1};
            assertThrows(IllegalArgumentException.class,()->new YoloDecoder.Workspace().decode(YoloDecoder.Values.floats(data),
                    shape,YoloDecoder.Family.YOLOV8,layout,2,true,320,320,320,320,320,320,0,0,.25f,.45f));
        }
    }
}
