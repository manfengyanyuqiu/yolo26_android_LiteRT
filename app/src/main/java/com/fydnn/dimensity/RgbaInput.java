package com.fydnn.dimensity;

import java.nio.ByteBuffer;

/** Packs an already resized/letterboxed RGBA frame, including OpenGL bottom-up readback. */
final class RgbaInput {
    static void pack(ByteBuffer rgba, int width, int height, int rowStride, boolean bottomUp,
                     float[] floats, byte[] bytes, float[] rgbLookup, byte[] quantLookup) {
        if (width <= 0 || height <= 0 || rowStride < (long)width*4
                || rgba.remaining() < (long)(height-1)*rowStride+(long)width*4
                || (floats == null) == (bytes == null)
                || (floats != null ? floats.length : bytes.length) != (long)width*height*3)
            throw new IllegalArgumentException("Invalid RGBA input buffer");
        int base=rgba.position(), k=0;
        for (int y=0; y<height; y++) {
            int row=base+(bottomUp ? height-1-y : y)*rowStride;
            if(floats!=null) {
                for(int x=0;x<width;x++) {
                    int pixel=row+x*4;floats[k++]=rgbLookup[rgba.get(pixel)&255];
                    floats[k++]=rgbLookup[rgba.get(pixel+1)&255];floats[k++]=rgbLookup[rgba.get(pixel+2)&255];
                }
            } else {
                for(int x=0;x<width;x++) {
                    int pixel=row+x*4;bytes[k++]=quantLookup[rgba.get(pixel)&255];
                    bytes[k++]=quantLookup[rgba.get(pixel+1)&255];bytes[k++]=quantLookup[rgba.get(pixel+2)&255];
                }
            }
        }
    }
    private RgbaInput() {}
}
