package com.fydnn.dimensity;

/** Clockwise sensor rotation followed by the same integer letterbox used for model input. */
public final class CameraGeometry {
    public static float[] vertices(int sourceWidth, int sourceHeight, int rotation, int targetWidth, int targetHeight) {
        if (sourceWidth <= 0 || sourceHeight <= 0 || targetWidth <= 0 || targetHeight <= 0
                || (rotation != 0 && rotation != 90 && rotation != 180 && rotation != 270))
            throw new IllegalArgumentException("Invalid camera geometry");
        int ow = rotation % 180 == 0 ? sourceWidth : sourceHeight;
        int oh = rotation % 180 == 0 ? sourceHeight : sourceWidth;
        float scale = Math.min((float)targetWidth / ow, (float)targetHeight / oh);
        int rw = Math.max(1, Math.round(ow * scale)), rh = Math.max(1, Math.round(oh * scale));
        int px = (targetWidth-rw)/2, py = (targetHeight-rh)/2;
        float cx = (2f*px+rw)/targetWidth-1f, cy = 1f-(2f*py+rh)/targetHeight;
        float[] corners = {-1,-1, 1,-1, -1,1, 1,1};
        for (int i=0; i<corners.length; i+=2) {
            float x = corners[i], y = corners[i+1];
            float rx = rotation == 0 ? x : rotation == 90 ? y : rotation == 180 ? -x : -y;
            float ry = rotation == 0 ? y : rotation == 90 ? -x : rotation == 180 ? -y : x;
            corners[i] = cx + rx * rw/targetWidth;
            corners[i+1] = cy + ry * rh/targetHeight;
        }
        return corners;
    }
    private CameraGeometry() {}
}
