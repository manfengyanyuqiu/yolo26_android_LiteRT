package com.fydnn.dimensity;

import java.util.Locale;

/** Filename conventions describe semantics; actual tensor dimensions determine layout. */
record ModelImportOptions(int familyIndex, int layoutIndex, boolean normalized) {
    static ModelImportOptions resolve(String filename, int[] outputShape,
                                      int selectedFamily, int classes) {
        String name = filename.toLowerCase(Locale.ROOT);
        int family = name.contains("yolo26") ? (name.contains("_e2e_") ? 3 : 2)
                : name.contains("yolov5") ? 0 : name.contains("yolov8") ? 1 : selectedFamily;
        int[] shape = YoloDecoder.decodedShape(outputShape);
        int channels = YoloDecoder.outputChannels(YoloDecoder.Family.values()[family], classes);
        boolean rows = shape[2] == channels, planes = shape[1] == channels;
        if (rows == planes)
            throw new IllegalArgumentException("输出张量无法确定布局，请检查模型系列和类别数："
                    + java.util.Arrays.toString(outputShape));
        // Compatibility with the exact names used in the first 320 model release.
        // Coordinate normalization cannot be inferred from tensor dimensions alone.
        boolean releaseName = name.matches("(?:yolov[58]|yolo26)[ns]_320_(?:fp16weights_float32io|int8)\\.tflite");
        return new ModelImportOptions(family, rows ? 0 : 1, name.contains("_norm_") || releaseName);
    }
}
