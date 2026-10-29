package com.fydnn.dimensity;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.reflect.Modifier;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Test;
import static org.junit.Assert.*;

public class PreparedNpuCallTest {
    private static final class Entry {
        long model; int signature, calls;
        long[] firstInput, firstOutput;
        RuntimeException failure;
        void run(long model, int signature, long[] input, long[] output) {
            if (failure != null) throw failure;
            this.model = model; this.signature = signature; calls++;
            if (firstInput == null) { firstInput = input; firstOutput = output; }
            else { assertSame(firstInput, input); assertSame(firstOutput, output); }
            assertArrayEquals(new long[]{22}, input);
            assertArrayEquals(new long[]{33}, output);
        }
        PreparedNpuCall call(Object[] resources) throws Exception {
            return new PreparedNpuCall(MethodHandles.lookup().findVirtual(Entry.class, "run",
                    MethodType.methodType(void.class, long.class, int.class, long[].class, long[].class))
                    .bindTo(this), 11, 22, 33, resources);
        }
    }

    @Test public void pinnedArtifactExposesExactNativeRunSignature() throws Exception {
        var api = PreparedNpuCall.bindings();
        assertEquals(void.class, api.run().getReturnType());
        assertTrue(Modifier.isNative(api.run().getModifiers()));
        assertTrue(Modifier.isStatic(api.run().getModifiers()));
        assertEquals(long.class, api.handle().getReturnType());
    }

    @Test public void repeatedCallsKeepModelSignatureAndBufferHandles() throws Exception {
        Entry entry = new Entry();
        try (var call = entry.call(new Object[]{new Object()})) {
            for (int i = 0; i < 100; i++) call.run();
            assertEquals(11, entry.model); assertEquals(0, entry.signature); assertEquals(100, entry.calls);
        }
    }

    @Test public void closedCallReleasesReferencesAndNeverUsesNativeHandles() throws Exception {
        Entry entry = new Entry(); Object[] resources = {new Object()};
        var call = entry.call(resources); call.close(); call.close();
        assertNull(resources[0]);
        assertThrows(IllegalStateException.class, call::run);
        assertEquals(0, entry.calls);
    }

    @Test public void otherWorkerCannotRunOrCloseOwnersCall() throws Exception {
        Entry entry = new Entry(); var call = entry.call(new Object[0]);
        AtomicReference<Throwable> runFailure = new AtomicReference<>(), closeFailure = new AtomicReference<>();
        Thread other = new Thread(() -> {
            try { call.run(); } catch (Throwable e) { runFailure.set(e); }
            try { call.close(); } catch (Throwable e) { closeFailure.set(e); }
        });
        other.start(); other.join();
        assertTrue(runFailure.get() instanceof IllegalStateException);
        assertTrue(closeFailure.get() instanceof IllegalStateException);
        call.run(); assertEquals(1, entry.calls); call.close();
    }

    @Test public void runtimeFailurePropagatesWithoutChangingExecutionPath() throws Exception {
        Entry entry = new Entry(); entry.failure = new IllegalStateException("driver error");
        try (var call = entry.call(new Object[0])) {
            assertSame(entry.failure, assertThrows(IllegalStateException.class, call::run));
            assertEquals(0, entry.calls);
        }
    }
}
