package com.fydnn.dimensity;

import org.junit.Test;
import java.util.List;
import static org.junit.Assert.*;

public class YoloDecoderTest {
    @Test public void quantizedOutputMatchesExplicitDequantizationInBothLayouts() {
        for (boolean unsigned : new boolean[]{false,true}) {
            byte[] raw={-78,-78,-108,-108, -28,52,-88};
            if (unsigned) for (int i=0;i<raw.length;i++) raw[i]=(byte)(raw[i]^0x80);
            int zero=unsigned?0:-128;
            float[] dequantized=new float[raw.length];
            for(int i=0;i<raw.length;i++)dequantized[i]=((unsigned?raw[i]&255:raw[i])-zero)*.005f;
            for (YoloDecoder.Layout layout:YoloDecoder.Layout.values()) {
                int[] shape=layout==YoloDecoder.Layout.ROWS?new int[]{1,1,7}:new int[]{1,7,1};
                var expected=run(dequantized,shape,YoloDecoder.Family.YOLOV5,layout,2,true);
                var actual=YoloDecoder.decode(YoloDecoder.Values.quantized(raw,.005f,zero,unsigned),shape,
                        YoloDecoder.Family.YOLOV5,layout,2,true,640,640,1280,720,640,360,0,140,.25f,.45f);
                assertEquals(expected,actual);assertEquals(1,actual.size());
            }
        }
    }
    @Test public void objectnessPruningPreservesThresholdBoundary() {
        assertEquals(1,run(new float[]{320,320,100,100,.25f,1f},new int[]{1,1,6},
                YoloDecoder.Family.YOLOV5,YoloDecoder.Layout.ROWS,1,false).size());
    }
    @Test public void mediatekRank4OutputKeepsBoxCoordinatesAndScores() {
        float[] data={320,320,100,100,.5f,.8f,.2f,100,200,50,60,.9f,.1f,.9f};
        var expected=run(data,new int[]{1,2,7},YoloDecoder.Family.YOLOV5,YoloDecoder.Layout.ROWS,2,false);
        assertEquals(expected,run(data,new int[]{1,1,2,7},YoloDecoder.Family.YOLOV5,YoloDecoder.Layout.ROWS,2,false));
        assertEquals(expected,run(data,new int[]{1,2,7,1},YoloDecoder.Family.YOLOV5,YoloDecoder.Layout.ROWS,2,false));
    }
    @Test(expected=IllegalArgumentException.class) public void nonSingletonRank4BatchRejected() {
        YoloDecoder.decodedShape(new int[]{1,2,3,7});
    }
    private List<YoloDecoder.Box> run(float[] data,int[] shape,YoloDecoder.Family family,
            YoloDecoder.Layout layout,int classes,boolean normalized){
        return YoloDecoder.decode(data,shape,family,layout,classes,normalized,640,640,
                1280,720,640,360,0,140,.25f,.45f);
    }
    @Test public void v5ObjectnessMultipliesClassScore(){
        float[] data={320,320,100,100,.5f,.8f,.2f};
        var boxes=run(data,new int[]{1,1,7},YoloDecoder.Family.YOLOV5,YoloDecoder.Layout.ROWS,2,false);
        assertEquals(1,boxes.size());var box=boxes.get(0);
        assertEquals(.4f,box.score(),.00001f);assertEquals(0,box.cls());
        assertEquals(540,box.left(),.001f);assertEquals(260,box.top(),.001f);
        assertEquals(740,box.right(),.001f);assertEquals(460,box.bottom(),.001f);
    }
    @Test public void v8ChannelsUsesScoresWithoutObjectness(){
        var boxes=run(new float[]{.5f,.5f,.25f,.25f,.1f,.9f},new int[]{1,6,1},
                YoloDecoder.Family.YOLOV8,YoloDecoder.Layout.CHANNELS,2,true);
        assertEquals(1,boxes.size());assertEquals(1,boxes.get(0).cls());
        assertEquals(.9f,boxes.get(0).score(),.00001f);assertEquals(480,boxes.get(0).left(),.001f);
    }
    @Test public void nmsSuppressesSameClassButPreservesOtherClasses(){
        float[] data={320,320,100,100,1,.9f,.1f,
                      321,321,100,100,1,.8f,.1f,
                      320,320,100,100,1,.1f,.85f};
        var boxes=run(data,new int[]{1,3,7},YoloDecoder.Family.YOLOV5,YoloDecoder.Layout.ROWS,2,false);
        assertEquals(2,boxes.size());assertEquals(0,boxes.get(0).cls());assertEquals(1,boxes.get(1).cls());
    }
    @Test public void lowObjectnessRemovesHighClassScore(){
        assertTrue(run(new float[]{320,320,100,100,.1f,.99f},new int[]{1,1,6},
            YoloDecoder.Family.YOLOV5,YoloDecoder.Layout.ROWS,1,false).isEmpty());
    }
    @Test(expected=IllegalArgumentException.class) public void wrongClassCountRejected(){
        run(new float[7],new int[]{1,1,7},YoloDecoder.Family.YOLOV5,YoloDecoder.Layout.ROWS,80,false);
    }
    @Test(expected=IllegalArgumentException.class) public void rawLogitsRejected(){
        run(new float[]{320,320,100,100,1,8},new int[]{1,1,6},
            YoloDecoder.Family.YOLOV5,YoloDecoder.Layout.ROWS,1,false);
    }
    @Test public void oddResizeUsesActualAxisScales(){
        var boxes=YoloDecoder.decode(new float[]{320,320,640,361,1,1},new int[]{1,1,6},
            YoloDecoder.Family.YOLOV5,YoloDecoder.Layout.ROWS,1,false,640,640,
            1280,721,640,361,0,139,.25f,.45f);
        assertEquals(1,boxes.size());assertEquals(1280,boxes.get(0).right(),.001f);
        assertEquals(721,boxes.get(0).bottom(),.001f);
    }
}
