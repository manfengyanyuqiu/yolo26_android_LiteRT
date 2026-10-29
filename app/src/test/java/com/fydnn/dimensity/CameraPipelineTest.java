package com.fydnn.dimensity;

import org.junit.Test;
import java.nio.ByteBuffer;
import java.util.List;
import static org.junit.Assert.*;

public class CameraPipelineTest {
    private CameraModeSelector.Mode mode(String id,int w,int h,boolean high,int low,int upper,long duration) {
        return new CameraModeSelector.Mode(id,w,h,90,high,low,upper,duration);
    }
    @Test public void selectsSmallestAdvertisedFixed120HighSpeedMode() {
        var selected=CameraModeSelector.select(List.of(
                mode("0",1920,1080,true,120,120,0),mode("0",1280,720,true,30,120,0),
                mode("1",1280,720,true,120,120,0),mode("0",640,480,false,120,120,8_000_000)));
        assertEquals("1",selected.cameraId()); assertEquals(1280,selected.width()); assertTrue(selected.highSpeed());
    }
    @Test(expected=IllegalStateException.class) public void variable120AndFixed240AreNotPretendedToBeFixed120() {
        CameraModeSelector.select(List.of(mode("0",1280,720,true,30,120,0),mode("0",1280,720,true,240,240,0)));
    }
    @Test public void regular120RequiresCompatibleFrameDuration() {
        var selected=CameraModeSelector.select(List.of(mode("0",640,480,false,120,120,16_666_667),
                mode("0",1280,720,false,120,120,8_333_333)));
        assertEquals(1280,selected.width()); assertFalse(selected.highSpeed());
    }
    @Test(expected=IllegalStateException.class) public void rejectsCameraWithOnly60Fps() {
        CameraModeSelector.select(List.of(mode("0",1280,720,false,60,60,16_666_667)));
    }
    @Test public void sensor90RotatesClockwiseAndLetterboxesWithoutStretching() {
        assertArrayEquals(new float[]{-.5625f,1,-.5625f,-1,.5625f,1,.5625f,-1},
                CameraGeometry.vertices(1280,720,90,224,224),.000001f);
        assertArrayEquals(new float[]{.5625f,-1,.5625f,1,-.5625f,-1,-.5625f,1},
                CameraGeometry.vertices(1280,720,270,224,224),.000001f);
    }
    @Test public void oddLetterboxPaddingMatchesTopLeftIntegerConvention() {
        assertArrayEquals(new float[]{-1,-.5f,1,-.5f,-1,.75f,1,.75f},
                CameraGeometry.vertices(3,2,0,8,8),.000001f);
    }
    @Test public void rgbaReadbackFlipsRowsPreservesRgbAndSkipsStridePadding() {
        ByteBuffer buffer=ByteBuffer.allocate(27);
        buffer.position(3);
        buffer.put(new byte[]{0,0,-1,-1, -1,-1,-1,-1, 99,99,99,99,
                -1,0,0,-1, 0,-1,0,-1, 99,99,99,99});
        buffer.position(3);
        float[] lookup=new float[256], actual=new float[12]; byte[] quant=new byte[256];
        for(int i=0;i<256;i++){lookup[i]=i/255f;quant[i]=(byte)(i-128);}
        RgbaInput.pack(buffer,2,2,12,true,actual,null,lookup,quant);
        assertArrayEquals(new float[]{1,0,0,0,1,0,0,0,1,1,1,1},actual,0);
        byte[] integers=new byte[12];
        RgbaInput.pack(buffer,2,2,12,true,null,integers,lookup,quant);
        assertArrayEquals(new byte[]{127,-128,-128,-128,127,-128,-128,-128,127,127,127,127},integers);
        assertEquals(3,buffer.position());
    }
    @Test(expected=IllegalArgumentException.class) public void truncatedRgbaBufferRejected() {
        RgbaInput.pack(ByteBuffer.allocate(15),2,2,8,true,new float[12],null,new float[256],new byte[256]);
    }
    @Test public void distinctMetersSeparateSensor120AndDetection30Fps() {
        FrameRateMeter sensor=new FrameRateMeter(), detector=new FrameRateMeter();
        for(int i=0;i<=120;i++)sensor.add(1+Math.round(i*1e9/120));
        sensor.add(1_000_000_001L); // Duplicate callback cannot inflate the count.
        for(int i=0;i<=30;i++)detector.add(1+Math.round(i*1e9/30));
        assertEquals(120,sensor.fps(),.001); assertEquals(30,detector.fps(),.001);
    }
}
