package com.fydnn.dimensity;

import org.junit.Test;
import static org.junit.Assert.*;

public class Yolo26DecoderTest {
    private java.util.List<YoloDecoder.Box> decode(float[] data, int[] shape, YoloDecoder.Family family,
            YoloDecoder.Layout layout, boolean normalized) {
        return YoloDecoder.decode(data, shape, family, layout, 2, normalized,
                320, 320, 640, 480, 320, 240, 0, 40, .25f, .45f);
    }

    @Test public void denseCoordinatesAndScoresMatchV8ConventionWithoutObjectness() {
        // Two nearby predictions, two classes, stored channel-major.
        float[] data = {.5f,.501f, .5f,.501f, .5f,.5f, .5f,.5f, .1f,.1f, .9f,.8f};
        var boxes = decode(data, new int[]{1,6,2,1}, YoloDecoder.Family.YOLO26,
                YoloDecoder.Layout.CHANNELS, true);
        assertEquals(1, boxes.size()); var box = boxes.get(0);
        assertEquals(1, box.cls()); assertEquals(.9f, box.score(), .00001f);
        assertEquals(160, box.left(), .001f); assertEquals(80, box.top(), .001f);
        assertEquals(480, box.right(), .001f); assertEquals(400, box.bottom(), .001f);
    }

    @Test public void sixColumnsUseXyxyAndClassIdAndDoNotApplySecondNms() {
        float[] rows = {80,80,240,240,.8f,1, 81,81,241,241,.9f,1, 0,0,0,0,0,-1};
        var boxes = decode(rows, new int[]{1,1,3,6}, YoloDecoder.Family.YOLO26_END2END,
                YoloDecoder.Layout.ROWS, false);
        assertEquals(2, boxes.size());
        assertEquals(.9f, boxes.get(0).score(), .00001f); assertEquals(1, boxes.get(0).cls());
        assertEquals(162, boxes.get(0).left(), .001f); assertEquals(82, boxes.get(0).top(), .001f);
        assertEquals(482, boxes.get(0).right(), .001f); assertEquals(402, boxes.get(0).bottom(), .001f);
    }

    @Test public void normalizedEndToEndSupportsChannelLayoutAndLetterbox() {
        var boxes = decode(new float[]{.25f,.25f,.75f,.75f,.9f,0}, new int[]{1,6,1},
                YoloDecoder.Family.YOLO26_END2END, YoloDecoder.Layout.CHANNELS, true);
        assertEquals(1, boxes.size());
        assertEquals(160, boxes.get(0).left(), .001f); assertEquals(80, boxes.get(0).top(), .001f);
        assertEquals(480, boxes.get(0).right(), .001f); assertEquals(400, boxes.get(0).bottom(), .001f);
    }

    @Test public void endToEndRejectsClassProbabilityMistakenForClassIndex() {
        assertThrows(IllegalArgumentException.class, () -> decode(new float[]{80,80,240,240,.9f,.8f},
                new int[]{1,1,6}, YoloDecoder.Family.YOLO26_END2END, YoloDecoder.Layout.ROWS, false));
        assertThrows(IllegalArgumentException.class, () -> decode(new float[]{80,80,240,240,.9f,2},
                new int[]{1,1,6}, YoloDecoder.Family.YOLO26_END2END, YoloDecoder.Layout.ROWS, false));
    }

    @Test public void denseInt8UsesCompactProbabilitiesAndSameBoxGeometry() {
        byte[] data = {-28,-28,-28,-28,-108,52};
        var boxes = YoloDecoder.decode(YoloDecoder.Values.quantized(data,.005f,-128,false),
                new int[]{1,6,1,1}, YoloDecoder.Family.YOLO26, YoloDecoder.Layout.CHANNELS, 2, true,
                320,320,640,480,320,240,0,40,.25f,.45f);
        assertEquals(1, boxes.size()); assertEquals(1, boxes.get(0).cls());
        assertEquals(.9f, boxes.get(0).score(), .00001f);
        assertEquals(160, boxes.get(0).left(), .001f); assertEquals(400, boxes.get(0).bottom(), .001f);
    }

    @Test public void sixColumnClassCountIsSeparateFromOutputWidth() {
        assertEquals(84, YoloDecoder.outputChannels(YoloDecoder.Family.YOLO26, 80));
        assertEquals(6, YoloDecoder.outputChannels(YoloDecoder.Family.YOLO26_END2END, 80));
        assertThrows(IllegalArgumentException.class, () -> YoloDecoder.outputChannels(YoloDecoder.Family.YOLO26, 0));
    }
}
