package com.fydnn.dimensity;

import java.util.Comparator;
import java.util.List;

/** Only modes explicitly advertising fixed 120 FPS are eligible. */
public final class CameraModeSelector {
    public record Mode(String cameraId, int width, int height, int sensorRotation,
                       boolean highSpeed, int minFps, int maxFps, long minFrameDurationNs) {}
    public static Mode select(List<Mode> modes) {
        return modes.stream().filter(m -> m.width > 0 && m.height > 0
                        && m.minFps == 120 && m.maxFps == 120
                        && (m.highSpeed || m.minFrameDurationNs == 0 || m.minFrameDurationNs <= 8_333_334L))
                .min(Comparator.<Mode>comparingInt(m -> m.highSpeed ? 0 : 1)
                        .thenComparingLong(m -> (long)m.width * m.height)
                        .thenComparing(Mode::cameraId))
                .orElseThrow(() -> new IllegalStateException(
                        "后置相机未公开可用的固定 120 FPS 模式。可见 FPS 范围：" + modes.stream()
                                .map(m -> m.cameraId+":"+(m.highSpeed?"高速":"普通")+"["+m.minFps+","+m.maxFps+"]")
                                .distinct().limit(24).collect(java.util.stream.Collectors.toList())));
    }
    private CameraModeSelector() {}
}
