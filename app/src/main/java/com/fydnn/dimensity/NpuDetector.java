package com.fydnn.dimensity;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.os.Build;
import android.util.Log;
import com.google.ai.edge.litert.Accelerator;
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider;
import com.google.ai.edge.litert.CompiledModel;
import com.google.ai.edge.litert.Environment;
import com.google.ai.edge.litert.LiteRtException;
import com.google.ai.edge.litert.TensorBuffer;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.nio.ByteBuffer;

/** All creation/run/close operations must use the same serial worker. */
public final class NpuDetector implements AutoCloseable {
    public record Result(List<YoloDecoder.Box> boxes, double preprocessMs, double inferenceMs,
                         double readbackMs, double postprocessMs, double scanMs, double nmsMs,
                         int candidates, int nmsComparisons) {}
    private record Spec(String name, int[] shape, DataType type, float scale, int zeroPoint) {
        static Spec of(Tensor t) {
            return new Spec(t.name(), t.shape(), t.dataType(),
                    t.quantizationParams().getScale(), t.quantizationParams().getZeroPoint());
        }
        void validate() {
            if (type != DataType.FLOAT32 && type != DataType.INT8 && type != DataType.UINT8)
                throw new IllegalArgumentException("Supported model I/O: FLOAT32 / INT8 / UINT8");
            if (type != DataType.FLOAT32 && !(scale > 0))
                throw new IllegalArgumentException("Invalid tensor quantization scale");
            for (int d : shape) if (d <= 0) throw new IllegalArgumentException("Export fixed tensor shapes");
        }
    }
    private Environment environment;
    private CompiledModel model;
    private PreparedNpuCall preparedCall;
    private TensorBuffer inputBuffer, outputBuffer;
    private Spec input, output;
    private List<TensorBuffer> inputs, outputs;
    private Bitmap resizedBitmap;
    private Canvas resizeCanvas;
    private final YoloDecoder.Workspace decoder = new YoloDecoder.Workspace();
    private final Paint resizePaint = new Paint(Paint.FILTER_BITMAP_FLAG);
    private final Rect resizeRect = new Rect();
    private int[] pixels;
    private float[] inputFloats;
    private byte[] inputBytes;
    private final float[] rgbLookup = new float[256];
    private final byte[] quantLookup = new byte[256];
    private final YoloDecoder.Family family;
    private final YoloDecoder.Layout layout;
    private final int classes;
    private final boolean normalized;
    public final int width, height;
    public final String description;

    public NpuDetector(Context context, File file, YoloDecoder.Family family,
            YoloDecoder.Layout layout, int classes, boolean normalized) throws LiteRtException {
        this.family = family; this.layout = layout; this.classes = classes; this.normalized = normalized;
        String stage = "model metadata";
        try {
            Log.i("DIMENSITY-NPU", "Loading " + file + " on " + Build.MODEL + " / " + Build.SOC_MODEL);
            if (!Build.SOC_MANUFACTURER.equalsIgnoreCase("Mediatek")
                    && !Build.SOC_MANUFACTURER.equalsIgnoreCase("MTK"))
                throw new IllegalStateException("This build requires a MediaTek device: " + Build.SOC_MANUFACTURER);
            // Metadata inspection only: no Interpreter.run() and no CPU inference attempt.
            try (Interpreter metadata = new Interpreter(file, new Interpreter.Options().setNumThreads(1))) {
                if (metadata.getInputTensorCount() != 1 || metadata.getOutputTensorCount() != 1)
                    throw new IllegalArgumentException("Export one RGB input and one decoded YOLO output, without NMS");
                input = Spec.of(metadata.getInputTensor(0)); output = Spec.of(metadata.getOutputTensor(0));
            }
            input.validate(); output.validate();
            if (input.shape.length != 4 || input.shape[0] != 1 || input.shape[3] != 3)
                throw new IllegalArgumentException("Expected fixed NHWC RGB input [1,H,W,3]");
            height = input.shape[1]; width = input.shape[2];
            int[] decodedShape = YoloDecoder.decodedShape(output.shape);
            int channels = YoloDecoder.outputChannels(family, classes);
            if (classes <= 0 || decodedShape[layout == YoloDecoder.Layout.ROWS ? 2 : 1] != channels)
                throw new IllegalArgumentException("Output shape " + Arrays.toString(output.shape)
                        + " does not match selected YOLO version/layout/class count");
            if (family == YoloDecoder.Family.YOLO26_END2END && output.type != DataType.FLOAT32)
                throw new IllegalArgumentException("YOLO26 six-column output requires FLOAT32 I/O (FP16 weights accepted); use dense output for INT8");
            String libDir = context.getApplicationInfo().nativeLibraryDir;
            for (String lib : new String[]{"libLiteRtCompilerPlugin_MediaTek.so", "libLiteRtDispatch_MediaTek.so"})
                if (!new File(libDir, lib).isFile()) throw new IllegalStateException("Missing NPU library: " + lib);
            stage = "MediaTek environment";
            environment = Environment.create(context, new BuiltinNpuAcceleratorProvider(context));
            if (!environment.getAvailableAccelerators().contains(Accelerator.NPU))
                throw new IllegalStateException("LiteRT did not expose an NPU. Check MediaTek driver and logcat.");
            stage = "NPU graph compilation";
            model = NpuOnlyModel.create(file.getAbsolutePath(), environment);
            stage = "NPU tensor buffers";
            // Metadata tensor names differ from SignatureDef aliases in SavedModel INT8 exports.
            // Bind in signature order, matching the one-input/one-output contract above.
            inputs = model.createInputBuffers(0);
            outputs = model.createOutputBuffers(0);
            if (inputs.size() != 1 || outputs.size() != 1)
                throw new IllegalArgumentException("Expected one input and output in model signature 0");
            inputBuffer = inputs.get(0); outputBuffer = outputs.get(0);
            preparedCall = PreparedNpuCall.create(model, inputBuffer, outputBuffer);
            resizedBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
            resizeCanvas = new Canvas(resizedBitmap);
            pixels = new int[width * height];
            if (input.type == DataType.FLOAT32) inputFloats = new float[pixels.length * 3];
            else inputBytes = new byte[pixels.length * 3];
            for (int i = 0; i < 256; i++) {
                rgbLookup[i] = i / 255f;
                if (input.type != DataType.FLOAT32) {
                    int min = input.type == DataType.INT8 ? -128 : 0;
                    quantLookup[i] = (byte) Math.max(min, Math.min(min + 255,
                            Math.round(rgbLookup[i] / input.scale) + input.zeroPoint));
                }
            }
            // Successful compilation alone does not guarantee driver execution.
            // Probe with a defined neutral frame and synchronize its output before
            // publishing a ready detector to image or camera inference.
            stage = "NPU first execution";
            if (inputFloats != null) {
                Arrays.fill(inputFloats, rgbLookup[114]); inputBuffer.writeFloat(inputFloats);
            } else {
                Arrays.fill(inputBytes, quantLookup[114]); inputBuffer.writeInt8(inputBytes);
            }
            preparedCall.run();
            Object probe = readOutput();
            if (probe instanceof float[] values) {
                for (float value : values)
                    if (!Float.isFinite(value)) throw new IllegalStateException("Non-finite NPU self-check output");
            }
            description = Build.SOC_MODEL + " / LiteRT 2.3.0 MediaTek / NPU only, first execution PASS\n"
                    + input.name + Arrays.toString(input.shape) + " " + input.type + " → "
                    + output.name + Arrays.toString(output.shape) + " " + output.type;
        } catch (LiteRtException | RuntimeException | Error e) {
            Log.e("DIMENSITY-NPU", "Load failed at " + stage + " on " + Build.SOC_MODEL, e);
            close(); throw e;
        }
    }

    public Result detect(Bitmap original) throws LiteRtException {
        long t0 = System.nanoTime();
        float scale = Math.min((float) width / original.getWidth(), (float) height / original.getHeight());
        int rw = Math.max(1, Math.round(original.getWidth() * scale));
        int rh = Math.max(1, Math.round(original.getHeight() * scale));
        int px = (width - rw) / 2, py = (height - rh) / 2;
        resizeCanvas.drawColor(Color.rgb(114, 114, 114));
        resizeRect.set(px, py, px + rw, py + rh);
        resizeCanvas.drawBitmap(original, null, resizeRect, resizePaint);
        resizedBitmap.getPixels(pixels, 0, width, 0, 0, width, height);
        int k = 0;
        if (inputFloats != null) {
            for(int pixel:pixels) { inputFloats[k++]=rgbLookup[(pixel>>>16)&255];
                inputFloats[k++]=rgbLookup[(pixel>>>8)&255];inputFloats[k++]=rgbLookup[pixel&255]; }
        } else {
            for(int pixel:pixels) { inputBytes[k++]=quantLookup[(pixel>>>16)&255];
                inputBytes[k++]=quantLookup[(pixel>>>8)&255];inputBytes[k++]=quantLookup[pixel&255]; }
        }
        if (inputFloats != null) inputBuffer.writeFloat(inputFloats); else inputBuffer.writeInt8(inputBytes);
        long t1 = System.nanoTime();
        return runPrepared(t0,t1,original.getWidth(),original.getHeight(),rw,rh,px,py);
    }

    /** Camera GPU has already rotated and letterboxed to width/height; no Bitmap or second resize. */
    public Result detectRgba(ByteBuffer rgba, int rowStride, boolean bottomUp) throws LiteRtException {
        long t0 = System.nanoTime();
        RgbaInput.pack(rgba,width,height,rowStride,bottomUp,inputFloats,inputBytes,rgbLookup,quantLookup);
        if (inputFloats != null) inputBuffer.writeFloat(inputFloats); else inputBuffer.writeInt8(inputBytes);
        return runPrepared(t0,System.nanoTime(),width,height,width,height,0,0);
    }

    private Result runPrepared(long t0, long t1, int originalWidth, int originalHeight,
                               int rw, int rh, int px, int py) throws LiteRtException {
        preparedCall.run();
        long t2 = System.nanoTime();
        YoloDecoder.Values data = output.type == DataType.FLOAT32
                ? YoloDecoder.Values.floats(outputBuffer.readFloat())
                : YoloDecoder.Values.quantized(outputBuffer.readInt8(), output.scale,
                        output.zeroPoint, output.type == DataType.UINT8);
        long t3 = System.nanoTime();
        List<YoloDecoder.Box> boxes = decoder.decode(data, output.shape, family, layout, classes,
                normalized, width, height, originalWidth, originalHeight, rw, rh, px, py, .25f, .45f);
        long t4 = System.nanoTime();
        return new Result(boxes, (t1-t0)/1e6, (t2-t1)/1e6, (t3-t2)/1e6, (t4-t3)/1e6,
                decoder.scanMs(),decoder.nmsMs(),decoder.candidates(),decoder.comparisons());
    }

    public double[] benchmark(Bitmap image, int warmup, int runs) throws LiteRtException {
        if (warmup < 0 || runs < 1) throw new IllegalArgumentException("Invalid run counts");
        for (int i=0; i<warmup; i++) detect(image);
        double[] times = new double[runs];
        for (int i=0; i<runs; i++) {
            long start = System.nanoTime();
            preparedCall.run();
            // Explicit readback completes any device work before stopping the timer.
            if (output.type == DataType.FLOAT32) outputBuffer.readFloat(); else outputBuffer.readInt8();
            times[i] = (System.nanoTime()-start)/1e6;
        }
        return statistics(times);
    }
    public double[] benchmarkEndToEnd(Bitmap image, int warmup, int runs) throws LiteRtException {
        if (warmup < 0 || runs < 1) throw new IllegalArgumentException("Invalid run counts");
        for (int i = 0; i < warmup; i++) detect(image);
        double[] times = new double[runs];
        for (int i = 0; i < runs; i++) {
            long start = System.nanoTime();
            detect(image);
            times[i] = (System.nanoTime() - start) / 1e6;
        }
        return statistics(times);
    }

    /** Same compiled model/input/output; alternates order to reduce warmup/clock bias. */
    public String benchmarkCalls(Bitmap image, int warmup, int runs) throws LiteRtException {
        if (warmup < 0 || runs < 1) throw new IllegalArgumentException("Invalid run counts");
        detect(image);
        for (int i = 0; i < warmup; i++) {
            model.run(inputs, outputs); readOutput(); preparedCall.run(); readOutput();
        }
        model.run(inputs, outputs); Object reference = readOutput();
        preparedCall.run(); Object actual = readOutput();
        boolean equal = reference instanceof float[] f ? Arrays.equals(f, (float[]) actual)
                : Arrays.equals((byte[]) reference, (byte[]) actual);
        if (!equal) throw new IllegalStateException("Prepared/public LiteRT outputs differ");
        double[] publicTimes = new double[runs], preparedTimes = new double[runs];
        for (int i = 0; i < runs; i++) {
            for (int j = 0; j < 2; j++) {
                boolean prepared = ((i + j) & 1) != 0;
                long start = System.nanoTime();
                if (prepared) preparedCall.run(); else model.run(inputs, outputs);
                readOutput(); // Include completion/readback in both measurements.
                (prepared ? preparedTimes : publicTimes)[i] = (System.nanoTime() - start) / 1e6;
            }
        }
        return "call_output_exact_match=" + equal
                + "\npublic_call_avg_p50_p95_ms=" + Arrays.toString(statistics(publicTimes))
                + "\nprepared_call_avg_p50_p95_ms=" + Arrays.toString(statistics(preparedTimes));
    }

    private Object readOutput() throws LiteRtException {
        return output.type == DataType.FLOAT32 ? outputBuffer.readFloat() : outputBuffer.readInt8();
    }
    private static double[] statistics(double[] times) {
        int runs = times.length;
        Arrays.sort(times);
        return new double[]{Arrays.stream(times).average().orElseThrow(),
                times[(runs-1)/2], times[(int)Math.ceil(runs*.95)-1]};
    }
    @Override public void close() {
        if (preparedCall != null) { preparedCall.close(); preparedCall = null; }
        if (resizedBitmap != null) { resizedBitmap.recycle(); resizedBitmap = null; resizeCanvas = null; }
        pixels = null; inputFloats = null; inputBytes = null; inputs = null; outputs = null;
        if (outputBuffer != null) { outputBuffer.close(); outputBuffer = null; }
        if (inputBuffer != null) { inputBuffer.close(); inputBuffer = null; }
        if (model != null) { model.close(); model = null; }
        if (environment != null) { environment.close(); environment = null; }
    }
}
