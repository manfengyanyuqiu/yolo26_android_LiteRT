package com.fydnn.dimensity;

import org.junit.Test;
import static org.junit.Assert.*;

public class ModelImportOptionsTest {
    @Test public void publishedModelsResetStaleFamilyLayoutAndNormalization() {
        String[] families={"yolov5","yolov8","yolo26"};
        for(int family=0;family<families.length;family++)
            for(String size:new String[]{"n","s"})
                for(String precision:new String[]{"fp16weights_float32io","int8"}) {
                    int[] shape=family==0?new int[]{1,1,6300,85}:new int[]{1,1,84,2100};
                    var options=ModelImportOptions.resolve(families[family]+size+"_320_"+precision+".tflite",shape,3,80);
                    assertEquals(family,options.familyIndex());
                    assertEquals(family==0?0:1,options.layoutIndex());
                    assertTrue(options.normalized());
                }
    }
    @Test public void tensorLayoutTakesPrecedenceOverFilenameAndRetainsCustomSemantics() {
        var options=ModelImportOptions.resolve("yolov8n_320_rows_norm_test.tflite",new int[]{1,1,84,2100},0,80);
        assertEquals(1,options.layoutIndex());assertTrue(options.normalized());
        options=ModelImportOptions.resolve("custom.tflite",new int[]{1,6300,7},0,2);
        assertEquals(0,options.familyIndex());assertEquals(0,options.layoutIndex());assertFalse(options.normalized());
    }
    @Test public void incompatibleAndAmbiguousShapesFailBeforeNpuCompilation() {
        assertThrows(IllegalArgumentException.class,()->ModelImportOptions.resolve("yolov8n.tflite",new int[]{1,1,84,2100},0,2));
        assertThrows(IllegalArgumentException.class,()->ModelImportOptions.resolve("yolov8n.tflite",new int[]{1,84,84},0,80));
    }
}
