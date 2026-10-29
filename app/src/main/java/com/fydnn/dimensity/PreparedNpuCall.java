package com.fydnn.dimensity;

import com.google.ai.edge.litert.CompiledModel;
import com.google.ai.edge.litert.JniHandle;
import com.google.ai.edge.litert.LiteRtException;
import com.google.ai.edge.litert.TensorBuffer;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;

/** Reuses the pinned LiteRT 2.3.0 JNI entry point and signature-0 buffer handles. */
final class PreparedNpuCall implements AutoCloseable {
    record Bindings(Method handle, Method run) {}

    static Bindings bindings() throws ReflectiveOperationException {
        Method handle = JniHandle.class.getDeclaredMethod(
                "getHandle$third_party_odml_litert_litert_kotlin_litert_api_for_maven");
        Method run = CompiledModel.class.getDeclaredMethod("nativeRun",
                long.class, int.class, long[].class, long[].class);
        run.setAccessible(true);
        return new Bindings(handle, run);
    }

    static PreparedNpuCall create(CompiledModel model, TensorBuffer input, TensorBuffer output) {
        try {
            Bindings api = bindings();
            return new PreparedNpuCall(MethodHandles.lookup().unreflect(api.run()),
                    (long) api.handle().invoke(model), (long) api.handle().invoke(input),
                    (long) api.handle().invoke(output), new Object[]{model, input, output});
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("LiteRT 2.3.0 prepared-run ABI mismatch", e);
        }
    }

    private final MethodHandle entry;
    private final long modelHandle;
    private final long[] inputs, outputs;
    // Keep borrowed native resources reachable. NpuDetector closes this call
    // before closing buffers/model; this adapter does not own those resources.
    private final Object[] resources;
    private final Thread owner = Thread.currentThread();
    private boolean closed;

    PreparedNpuCall(MethodHandle entry, long model, long input, long output, Object[] resources) {
        if (model == 0 || input == 0 || output == 0) throw new IllegalArgumentException("Null LiteRT handle");
        this.entry = entry; modelHandle = model;
        inputs = new long[]{input}; outputs = new long[]{output}; this.resources = resources;
    }

    void run() throws LiteRtException {
        requireOwner();
        if (closed) throw new IllegalStateException("Prepared NPU call is closed");
        try {
            entry.invokeExact(modelHandle, 0, inputs, outputs);
        } catch (LiteRtException e) { throw e;
        } catch (RuntimeException | Error e) { throw e;
        } catch (Throwable e) { throw new IllegalStateException("Prepared NPU call failed", e); }
    }

    private void requireOwner() {
        if (Thread.currentThread() != owner) throw new IllegalStateException("NPU call must use its creating worker");
    }

    @Override public void close() {
        requireOwner(); closed = true;
        java.util.Arrays.fill(resources, null);
    }
}
