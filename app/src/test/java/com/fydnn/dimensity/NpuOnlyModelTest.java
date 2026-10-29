package com.fydnn.dimensity;

import com.google.ai.edge.litert.CompiledModel;
import java.lang.reflect.Modifier;
import org.junit.Test;
import static org.junit.Assert.*;

public class NpuOnlyModelTest {
    @Test public void publishedArtifactHasRequiredNpuOnlyAbi() throws Exception {
        var api = NpuOnlyModel.bindings();
        assertEquals(long.class, api.create().getReturnType());
        assertTrue(Modifier.isStatic(api.create().getModifiers()));
        assertTrue(Modifier.isNative(api.create().getModifiers()));
        assertEquals(long.class, api.environmentHandle().getReturnType());
        assertEquals(CompiledModel.class, api.wrap().getDeclaringClass());
        assertTrue(Modifier.isNative(api.destroy().getModifiers()));
    }
}
