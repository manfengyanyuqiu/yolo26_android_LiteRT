package com.fydnn.dimensity;

import android.app.Activity;
import android.app.Instrumentation;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import com.google.ai.edge.litert.TensorBuffer;
import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;

/** Device integration check against the app's real NPU-only path, no test framework dependency. */
public final class NpuDeviceInstrumentation extends Instrumentation {
    private Bundle arguments;
    @Override public void onCreate(Bundle arguments) { this.arguments = arguments; start(); }
    @Override public void onStart() {
        Bundle result = new Bundle();
        File report = new File(getTargetContext().getFilesDir(), "device-test.txt");
        try (PrintWriter writer = new PrintWriter(report)) {
            Thread testThread=Thread.currentThread();
            Thread watchdog=new Thread(()->{
                try {Thread.sleep(12000);if(!testThread.isAlive())return;
                    File trace=new File(getTargetContext().getFilesDir(),"latency-stack.txt");
                    try(PrintWriter dump=new PrintWriter(trace)) {
                        Thread.getAllStackTraces().forEach((thread,stack)->{
                            dump.println(thread.getName()+" "+thread.getState());
                            for(var frame:stack)dump.println("  "+frame);
                        });
                    }
                }catch(Exception ignored){}
            },"DIMENSITY-test-watchdog");watchdog.setDaemon(true);watchdog.start();
            File model = new File(getTargetContext().getFilesDir(), arguments.getString("model", "detector.tflite"));
            try (NpuDetector detector = new NpuDetector(getTargetContext(), model,
                    YoloDecoder.Family.valueOf(arguments.getString("family", "YOLOV5")),
                    YoloDecoder.Layout.valueOf(arguments.getString("layout", "ROWS")),
                    Integer.parseInt(arguments.getString("classes", "80")),
                    Boolean.parseBoolean(arguments.getString("normalized", "false")))) {
                writer.println(detector.description); writer.flush();
                String imageName = arguments.getString("image");
                if (imageName != null) {
                    Bitmap image = BitmapFactory.decodeFile(new File(getTargetContext().getFilesDir(), imageName).getPath());
                    if (image == null) throw new IllegalArgumentException("Device test image missing");
                    try {
                        writer.println("stage=detect_image "+image.getWidth()+"x"+image.getHeight());writer.flush();
                        var detection = detector.detect(image);
                        writer.println("detection=" + detection);
                        writer.flush();
                        if(Boolean.parseBoolean(arguments.getString("pipelineBenchmark","false"))) {
                            try(Interpreter metadata=new Interpreter(model,new Interpreter.Options().setNumThreads(1))) {
                                var spec=metadata.getOutputTensor(0);
                                writer.println(PipelineDeviceBenchmark.run(detector,java.nio.file.Files.readAllBytes(
                                        new File(getTargetContext().getFilesDir(),imageName).toPath()),spec.shape(),
                                        spec.dataType()==DataType.FLOAT32,spec.quantizationParams().getScale(),
                                        spec.quantizationParams().getZeroPoint(),spec.dataType()==DataType.UINT8,
                                        YoloDecoder.Family.valueOf(arguments.getString("family","YOLOV5")),
                                        YoloDecoder.Layout.valueOf(arguments.getString("layout","ROWS")),
                                        Integer.parseInt(arguments.getString("classes","80")),
                                        Boolean.parseBoolean(arguments.getString("normalized","false"))));writer.flush();
                            }
                        }
                        if(Boolean.parseBoolean(arguments.getString("decoderBenchmark","false"))) {
                            try(Interpreter metadata=new Interpreter(model,new Interpreter.Options().setNumThreads(1))) {
                                Field field=NpuDetector.class.getDeclaredField("outputBuffer");field.setAccessible(true);
                                TensorBuffer tensor=(TensorBuffer)field.get(detector);var spec=metadata.getOutputTensor(0);
                                writer.println(DecoderDeviceBenchmark.run(spec.dataType()==DataType.FLOAT32?tensor.readFloat():null,
                                        spec.dataType()==DataType.FLOAT32?null:tensor.readInt8(),
                                        spec.quantizationParams().getScale(),spec.quantizationParams().getZeroPoint(),spec.dataType()==DataType.UINT8,
                                        spec.shape(),YoloDecoder.Family.valueOf(arguments.getString("family","YOLOV5")),
                                        YoloDecoder.Layout.valueOf(arguments.getString("layout","ROWS")),
                                        Integer.parseInt(arguments.getString("classes","80")),
                                        Boolean.parseBoolean(arguments.getString("normalized","false")),
                                        detector.width,detector.height,image.getWidth(),image.getHeight()));writer.flush();
                            }
                        }
                        // Export the actual input and output for host numerical comparison.
                        try (Interpreter metadata = new Interpreter(model, new Interpreter.Options().setNumThreads(1))) {
                        for (String name : new String[]{"inputBuffer", "outputBuffer"}) {
                            Field field = NpuDetector.class.getDeclaredField(name); field.setAccessible(true);
                            TensorBuffer tensor = (TensorBuffer) field.get(detector);
                            var spec = name.equals("inputBuffer") ? metadata.getInputTensor(0) : metadata.getOutputTensor(0);
                            byte[] raw;
                            if (spec.dataType() == DataType.FLOAT32) {
                                float[] data = tensor.readFloat();
                                ByteBuffer bytes = ByteBuffer.allocate(data.length * 4).order(ByteOrder.LITTLE_ENDIAN);
                                bytes.asFloatBuffer().put(data); raw = bytes.array();
                            } else raw = tensor.readInt8();
                            writer.println(name + "=" + spec.dataType() + " scale=" + spec.quantizationParams().getScale()
                                    + " zero=" + spec.quantizationParams().getZeroPoint());
                            try (FileOutputStream out = new FileOutputStream(new File(getTargetContext().getFilesDir(), name + ".bin"))) {
                                out.write(raw);
                            }
                        }
                        }
                        if (Boolean.parseBoolean(arguments.getString("benchmark", "true"))) {
                            writer.println("benchmark_avg_p50_p95_ms=" + Arrays.toString(detector.benchmark(image, 5, 50)));
                            writer.println("image_end_to_end_avg_p50_p95_ms=" + Arrays.toString(detector.benchmarkEndToEnd(image, 5, 50)));
                        }
                        if (Boolean.parseBoolean(arguments.getString("callBenchmark", "false")))
                            writer.println(detector.benchmarkCalls(image, 20, 200));
                    } finally { image.recycle(); }
                }
                writer.println("PASS"); writer.flush();
                result.putString("stream", "NPU compile/run PASS; report: " + report);
                finish(Activity.RESULT_OK, result);
            }
        } catch (Throwable e) {
            Log.e("DIMENSITY-NPU", "Device integration check failed", e);
            try (PrintWriter writer = new PrintWriter(report)) { e.printStackTrace(writer); }
            catch (Exception writeError) { e.addSuppressed(writeError); }
            result.putString("stream", Log.getStackTraceString(e)); finish(Activity.RESULT_CANCELED, result);
        }
    }
}
