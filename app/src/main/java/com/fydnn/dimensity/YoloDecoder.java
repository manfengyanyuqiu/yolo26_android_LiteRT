package com.fydnn.dimensity;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Arrays;

/** Single decoded output only; raw multi-head output must be decoded at export. */
public final class YoloDecoder {
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
        return new Workspace().decode(data,shape,family,layout,classCount,normalized,inputWidth,inputHeight,
                originalWidth,originalHeight,resizedWidth,resizedHeight,padX,padY,confidence,iouThreshold);
    }
    /** One workspace per serial detector; mutable scratch never escapes to the renderer. */
    public static final class Workspace {
        private float[] bestScore=new float[0], objectness=new float[0], left=new float[0], top=new float[0],
                right=new float[0], bottom=new float[0], score=new float[0];
        private int[] bestClass=new int[0], label=new int[0];
        private final int[] kept=new int[300], nextKept=new int[300];
        private int[] classHeads=new int[0];
        private long[] order=new long[0];
        private final float[] quantized=new float[256];
        private float quantScale;
        private int quantZero;
        private boolean quantUnsigned;
        private double scanMs, nmsMs;
        private int candidateCount, comparisonCount;
        public double scanMs() { return scanMs; }
        public double nmsMs() { return nmsMs; }
        public int candidates() { return candidateCount; }
        public int comparisons() { return comparisonCount; }
        private void capacity(int count,int classes) {
            if(bestScore.length<count) {
                bestScore=new float[count];objectness=new float[count];bestClass=new int[count];
                left=new float[count];top=new float[count];right=new float[count];bottom=new float[count];
                score=new float[count];label=new int[count];order=new long[count];
            }
            if(classHeads.length<classes)classHeads=new int[classes];
        }
        private void quantization(Values data) {
            if(data.bytes==null || (quantScale==data.scale && quantZero==data.zero && quantUnsigned==data.unsigned))return;
            quantScale=data.scale;quantZero=data.zero;quantUnsigned=data.unsigned;
            for(int q=0;q<256;q++)quantized[q]=((data.unsigned?q:q-128)-quantZero)*quantScale;
        }
        private float value(Values data,int index) {
            if(data.floats!=null)return data.floats[index];
            return quantized[data.unsigned ? data.bytes[index]&255 : data.bytes[index]+128];
        }
        public List<Box> decode(Values data,int[] shape,Family family,Layout layout,int classCount,
                boolean normalized,int inputWidth,int inputHeight,int originalWidth,int originalHeight,
                int resizedWidth,int resizedHeight,int padX,int padY,float confidence,float iouThreshold) {
        long start=System.nanoTime();
        shape = decodedShape(shape);
        if (shape.length != 3 || shape[0] != 1 || classCount <= 0)
            throw new IllegalArgumentException("Expected [1,N,C] or [1,C,N] and positive class count");
        int channels = outputChannels(family, classCount);
        int actualChannels = layout == Layout.ROWS ? shape[2] : shape[1];
        int count = layout == Layout.ROWS ? shape[1] : shape[2];
        if (actualChannels != channels || count <= 0 || (long) count * channels != data.length())
            throw new IllegalArgumentException("Output shape does not match selected family/layout/classes");
        if (family == Family.YOLO26_END2END) {
            var boxes=endToEnd(data, layout, count, classCount, normalized, inputWidth, inputHeight,
                    originalWidth, originalHeight, resizedWidth, resizedHeight, padX, padY, confidence);
            scanMs=(System.nanoTime()-start)/1e6;nmsMs=0;candidateCount=boxes.size();comparisonCount=0;
            return boxes;
        }
        capacity(count,classCount);quantization(data);
        int classOffset = family == Family.YOLOV5 ? 5 : 4;
        boolean v5=family==Family.YOLOV5;
        if(layout==Layout.CHANNELS) {
            Arrays.fill(bestScore,0,count,-1);Arrays.fill(bestClass,0,count,0);
            for(int i=0;i<count;i++) {
                float obj=v5?value(data,4*count+i):1;probability(obj);objectness[i]=obj;
            }
            // Class planes are contiguous. Avoid jumping count elements for every class read.
            if(!v5 && data.floats!=null) {
                float[] values=data.floats;
                for(int c=0;c<classCount;c++) {
                    int base=(classOffset+c)*count;
                    for(int i=0;i<count;i++) {
                        float p=values[base+i];
                        if(!(p>=-.01f && p<=1.01f))probability(p);
                        if(p>bestScore[i]) {bestScore[i]=p;bestClass[i]=c;}
                    }
                }
            } else {
                for(int c=0;c<classCount;c++) {
                    int base=(classOffset+c)*count;
                    for(int i=0;i<count;i++) {
                        if(v5 && objectness[i]*1.01f<confidence)continue;
                        float p=value(data,base+i);probability(p);
                        if(p>bestScore[i]) { bestScore[i]=p;bestClass[i]=c; }
                    }
                }
            }
        } else {
            for(int i=0;i<count;i++) {
                int base=i*channels;
                float obj=v5?value(data,base+4):1;probability(obj);objectness[i]=obj;
                bestScore[i]=-1;bestClass[i]=0;
                if(obj*1.01f<confidence)continue;
                for(int c=0;c<classCount;c++) {
                    float p=value(data,base+classOffset+c);probability(p);
                    if(p>bestScore[i]) { bestScore[i]=p;bestClass[i]=c; }
                }
            }
        }
        float sx=(float)originalWidth/resizedWidth,sy=(float)originalHeight/resizedHeight;
        int candidates=0;
        for (int i = 0; i < count; i++) {
            if(objectness[i]*1.01f<confidence)continue;
            float confidenceScore=objectness[i]*bestScore[i];
            if(confidenceScore<confidence)continue;
            int base=layout==Layout.ROWS?i*channels:i,step=layout==Layout.ROWS?1:count;
            float cx=value(data,base),cy=value(data,base+step),w=value(data,base+2*step),h=value(data,base+3*step);
            if (!Float.isFinite(cx) || !Float.isFinite(cy) || !Float.isFinite(w) || !Float.isFinite(h))
                throw new IllegalArgumentException("Non-finite model output");
            if (normalized) { cx *= inputWidth; w *= inputWidth; cy *= inputHeight; h *= inputHeight; }
            if (w <= 0 || h <= 0) continue;
            float x1 = clamp((cx - w / 2 - padX) * sx, originalWidth);
            float y1 = clamp((cy - h / 2 - padY) * sy, originalHeight);
            float x2 = clamp((cx + w / 2 - padX) * sx, originalWidth);
            float y2 = clamp((cy + h / 2 - padY) * sy, originalHeight);
            if(x2<=x1 || y2<=y1)continue;
            left[candidates]=x1;top[candidates]=y1;right[candidates]=x2;bottom[candidates]=y2;
            score[candidates]=confidenceScore;label[candidates]=bestClass[i];
            int bits=Float.floatToIntBits(confidenceScore),sortable=bits^((bits>>31)&0x7fffffff);
            // JDK primitive sort, descending score and stable original index for ties.
            order[candidates]=((long)~sortable<<32)|(candidates&0xffffffffL);candidates++;
        }
        long scanned=System.nanoTime();
        Arrays.sort(order,0,candidates);Arrays.fill(classHeads,0,classCount,-1);
        int retained=0,comparisons=0;
        for(int n=0;n<candidates;n++) {
            int candidate=(int)order[n],cls=label[candidate];boolean suppressed=false;
            for(int slot=classHeads[cls];slot!=-1;slot=nextKept[slot]) {
                int previous=kept[slot];comparisons++;
                float intersection=Math.max(0,Math.min(right[previous],right[candidate])-Math.max(left[previous],left[candidate]))
                        *Math.max(0,Math.min(bottom[previous],bottom[candidate])-Math.max(top[previous],top[candidate]));
                float union=(right[previous]-left[previous])*(bottom[previous]-top[previous])
                        +(right[candidate]-left[candidate])*(bottom[candidate]-top[candidate])-intersection;
                if((union>0?intersection/union:0)>iouThreshold) { suppressed=true;break; }
            }
            if(!suppressed) {
                kept[retained]=candidate;nextKept[retained]=classHeads[cls];classHeads[cls]=retained;
                if(++retained==300)break;
            }
        }
        List<Box> result=new ArrayList<>(retained);
        for(int n=0;n<retained;n++) {
            int i=kept[n];result.add(new Box(left[i],top[i],right[i],bottom[i],score[i],label[i]));
        }
        scanMs=(scanned-start)/1e6;nmsMs=(System.nanoTime()-scanned)/1e6;
        candidateCount=candidates;comparisonCount=comparisons;return result;
        }
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
        // Ordered comparisons also reject NaN and infinities without Float.isFinite's extra call chain.
        if (!(p >= -0.01f && p <= 1.01f))
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
