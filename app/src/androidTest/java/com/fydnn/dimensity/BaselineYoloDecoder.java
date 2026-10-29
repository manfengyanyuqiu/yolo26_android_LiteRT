package com.fydnn.dimensity;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/** Single decoded output only; raw multi-head output must be decoded at export. */
public final class BaselineYoloDecoder {
    public enum Family { YOLOV5, YOLOV8, YOLO26, YOLO26_END2END }
    public enum Layout { ROWS, CHANNELS }
    public record Box(float left, float top, float right, float bottom, float score, int cls) {}
    public static int outputChannels(Family family, int classes) {
        if (classes <= 0) throw new IllegalArgumentException("Class count must be positive");
        return family == Family.YOLO26_END2END ? 6 : classes + (family == Family.YOLOV5 ? 5 : 4);
    }
    /** Quantized output stays in its compact readback array; dequantize only accessed elements. */
    public static final class Values {
        private final float[] floats;
        private final byte[] bytes;
        private final float scale;
        private final int zero;
        private final boolean unsigned;
        private Values(float[] floats, byte[] bytes, float scale, int zero, boolean unsigned) {
            this.floats=floats; this.bytes=bytes; this.scale=scale; this.zero=zero; this.unsigned=unsigned;
        }
        public static Values floats(float[] data) { return new Values(data,null,0,0,false); }
        public static Values quantized(byte[] data, float scale, int zero, boolean unsigned) {
            if (!(scale > 0) || !Float.isFinite(scale)) throw new IllegalArgumentException("Invalid quantization scale");
            return new Values(null,data,scale,zero,unsigned);
        }
        int length() { return floats != null ? floats.length : bytes.length; }
        float get(int i) { return floats != null ? floats[i] : ((unsigned ? bytes[i]&255 : bytes[i])-zero)*scale; }
    }
    public static int[] decodedShape(int[] shape) {
        if (shape.length == 4 && shape[0] == 1) {
            if (shape[1] == 1) return new int[]{1, shape[2], shape[3]};
            if (shape[3] == 1) return new int[]{1, shape[1], shape[2]};
        }
        if (shape.length == 3 && shape[0] == 1) return shape.clone();
        throw new IllegalArgumentException("Expected [1,N,C]/[1,C,N], optionally padded to rank 4");
    }
    public static List<Box> decode(float[] data, int[] shape, Family family, Layout layout,
            int classCount, boolean normalized, int inputWidth, int inputHeight,
            int originalWidth, int originalHeight, int resizedWidth, int resizedHeight,
            int padX, int padY, float confidence, float iouThreshold) {
        return decode(Values.floats(data),shape,family,layout,classCount,normalized,inputWidth,inputHeight,
                originalWidth,originalHeight,resizedWidth,resizedHeight,padX,padY,confidence,iouThreshold);
    }
    public static List<Box> decode(Values data, int[] shape, Family family, Layout layout,
            int classCount, boolean normalized, int inputWidth, int inputHeight,
            int originalWidth, int originalHeight, int resizedWidth, int resizedHeight,
            int padX, int padY, float confidence, float iouThreshold) {
        shape = decodedShape(shape);
        if (shape.length != 3 || shape[0] != 1 || classCount <= 0)
            throw new IllegalArgumentException("Expected [1,N,C] or [1,C,N] and positive class count");
        int channels = outputChannels(family, classCount);
        int actualChannels = layout == Layout.ROWS ? shape[2] : shape[1];
        int count = layout == Layout.ROWS ? shape[1] : shape[2];
        if (actualChannels != channels || count <= 0 || (long) count * channels != data.length())
            throw new IllegalArgumentException("Output shape does not match selected family/layout/classes");
        if (family == Family.YOLO26_END2END)
            return endToEnd(data, layout, count, classCount, normalized, inputWidth, inputHeight,
                    originalWidth, originalHeight, resizedWidth, resizedHeight, padX, padY, confidence);
        List<Box> candidates = new ArrayList<>();
        int classOffset = family == Family.YOLOV5 ? 5 : 4;
        for (int i = 0; i < count; i++) {
            float objectness = family == Family.YOLOV5 ? at(data, i, 4, count, channels, layout) : 1;
            probability(objectness);
            // Valid class probabilities are at most 1.01 (the accepted numerical tolerance).
            if (objectness * 1.01f < confidence) continue;
            int best = 0; float bestScore = -1;
            for (int c = 0; c < classCount; c++) {
                float p = at(data, i, classOffset + c, count, channels, layout);
                probability(p);
                if (p > bestScore) { bestScore = p; best = c; }
            }
            float score = objectness * bestScore;
            if (score < confidence) continue;
            float cx = at(data, i, 0, count, channels, layout);
            float cy = at(data, i, 1, count, channels, layout);
            float w = at(data, i, 2, count, channels, layout);
            float h = at(data, i, 3, count, channels, layout);
            if (!Float.isFinite(cx) || !Float.isFinite(cy) || !Float.isFinite(w) || !Float.isFinite(h))
                throw new IllegalArgumentException("Non-finite model output");
            if (normalized) { cx *= inputWidth; w *= inputWidth; cy *= inputHeight; h *= inputHeight; }
            if (w <= 0 || h <= 0) continue;
            float sx = (float) originalWidth / resizedWidth, sy = (float) originalHeight / resizedHeight;
            float x1 = clamp((cx - w / 2 - padX) * sx, originalWidth);
            float y1 = clamp((cy - h / 2 - padY) * sy, originalHeight);
            float x2 = clamp((cx + w / 2 - padX) * sx, originalWidth);
            float y2 = clamp((cy + h / 2 - padY) * sy, originalHeight);
            if (x2 > x1 && y2 > y1) candidates.add(new Box(x1, y1, x2, y2, score, best));
        }
        candidates.sort(Comparator.comparingDouble(Box::score).reversed());
        List<Box> kept = new ArrayList<>();
        for (Box candidate : candidates) {
            boolean suppressed = false;
            for (Box previous : kept) {
                if (previous.cls == candidate.cls && iou(previous, candidate) > iouThreshold) {
                    suppressed = true; break;
                }
            }
            if (!suppressed) kept.add(candidate);
            if (kept.size() == 300) break;
        }
        return kept;
    }

    /** YOLO26 one-to-one output is xyxy, score, class-id; no objectness or second NMS. */
    private static List<Box> endToEnd(Values data, Layout layout, int count, int classes,
            boolean normalized, int inputWidth, int inputHeight, int originalWidth, int originalHeight,
            int resizedWidth, int resizedHeight, int padX, int padY, float confidence) {
        List<Box> boxes = new ArrayList<>();
        float sx = (float) originalWidth / resizedWidth, sy = (float) originalHeight / resizedHeight;
        for (int i = 0; i < count; i++) {
            float score = at(data, i, 4, count, 6, layout);
            probability(score);
            if (score < confidence) continue;
            float label = at(data, i, 5, count, 6, layout);
            int cls = Math.round(label);
            if (!Float.isFinite(label) || cls < 0 || cls >= classes || Math.abs(label - cls) > .001f)
                throw new IllegalArgumentException("Expected integer YOLO26 class index within class count");
            float x1 = at(data, i, 0, count, 6, layout), y1 = at(data, i, 1, count, 6, layout);
            float x2 = at(data, i, 2, count, 6, layout), y2 = at(data, i, 3, count, 6, layout);
            if (!Float.isFinite(x1) || !Float.isFinite(y1) || !Float.isFinite(x2) || !Float.isFinite(y2))
                throw new IllegalArgumentException("Non-finite YOLO26 box coordinates");
            if (normalized) { x1 *= inputWidth; x2 *= inputWidth; y1 *= inputHeight; y2 *= inputHeight; }
            x1 = clamp((x1 - padX) * sx, originalWidth); x2 = clamp((x2 - padX) * sx, originalWidth);
            y1 = clamp((y1 - padY) * sy, originalHeight); y2 = clamp((y2 - padY) * sy, originalHeight);
            if (x2 > x1 && y2 > y1) boxes.add(new Box(x1, y1, x2, y2, score, cls));
        }
        boxes.sort(Comparator.comparingDouble(Box::score).reversed());
        return boxes.size() > 300 ? new ArrayList<>(boxes.subList(0, 300)) : boxes;
    }
    private static float at(Values d, int n, int c, int count, int channels, Layout layout) {
        return d.get(layout == Layout.ROWS ? n * channels + c : c * count + n);
    }
    private static void probability(float p) {
        if (!Float.isFinite(p) || p < -0.01f || p > 1.01f)
            throw new IllegalArgumentException("Expected decoded probabilities; export raw logits with decode");
    }
    private static float clamp(float x, int limit) { return Math.max(0, Math.min(x, limit)); }
    private static float iou(Box a, Box b) {
        float intersection = Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left))
                * Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top));
        float union = (a.right - a.left) * (a.bottom - a.top)
                + (b.right - b.left) * (b.bottom - b.top) - intersection;
        return union > 0 ? intersection / union : 0;
    }
}

