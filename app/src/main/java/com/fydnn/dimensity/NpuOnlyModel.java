package com.fydnn.dimensity;

import com.google.ai.edge.litert.Accelerator;
import com.google.ai.edge.litert.CompiledModel;
import com.google.ai.edge.litert.Environment;
import com.google.ai.edge.litert.JniHandle;
import com.google.ai.edge.litert.LiteRtException;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

/** Adapter for the pinned LiteRT 2.3.0 ABI. Its Java factory adds CPU to NPU. */
final class NpuOnlyModel {
    record Bindings(Method environmentHandle, Method create, Method destroy,
                    Constructor<CompiledModel> wrap) {}

    // Resolving members does not initialize the native runtime, so this contract
    // is also checked against the actual Gradle dependency by a host unit test.
    static Bindings bindings() throws ReflectiveOperationException {
        Method handle = JniHandle.class.getDeclaredMethod(
                "getHandle$third_party_odml_litert_litert_kotlin_litert_api_for_maven");
        Method create = CompiledModel.class.getDeclaredMethod("nativeCreateFromFile",
                long.class, String.class, int[].class, int[].class, String[].class,
                int[].class, String[].class, int[].class, String[].class);
        Method destroy = CompiledModel.class.getDeclaredMethod("nativeDestroy", long.class);
        Constructor<CompiledModel> wrap = CompiledModel.class.getDeclaredConstructor(
                long.class, Environment.class, boolean.class);
        create.setAccessible(true);
        destroy.setAccessible(true);
        wrap.setAccessible(true);
        return new Bindings(handle, create, destroy, wrap);
    }

    static CompiledModel create(String path, Environment environment) throws LiteRtException {
        long nativeHandle = 0;
        Bindings api = null;
        try {
            api = bindings();
            long environmentHandle = (long) api.environmentHandle().invoke(environment);
            nativeHandle = (long) api.create().invoke(null, environmentHandle, path,
                    new int[]{Accelerator.NPU.getValue()},
                    new int[0], new String[0], new int[0], new String[0], new int[0], new String[0]);
            if (nativeHandle == 0) throw new IllegalStateException("LiteRT returned a null model handle");
            CompiledModel model = api.wrap().newInstance(nativeHandle, environment, false);
            nativeHandle = 0; // CompiledModel now owns the handle; caller owns environment.
            return model;
        } catch (InvocationTargetException e) {
            Throwable cause = e.getCause();
            if (cause instanceof LiteRtException error) throw error;
            if (cause instanceof RuntimeException error) throw error;
            if (cause instanceof Error error) throw error;
            throw new IllegalStateException("LiteRT NPU creation failed", cause);
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("LiteRT 2.3.0 NPU-only ABI mismatch", e);
        } finally {
            if (nativeHandle != 0 && api != null) {
                try { api.destroy().invoke(null, nativeHandle); }
                catch (ReflectiveOperationException e) {
                    throw new IllegalStateException("Failed to release LiteRT model", e);
                }
            }
        }
    }
}
