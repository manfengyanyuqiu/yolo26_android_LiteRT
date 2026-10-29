package com.fydnn.dimensity;

/** Rate from monotonic timestamps, independently usable for sensor, consumed frames and results. */
final class FrameRateMeter {
    private long first,last,lastArrival;
    private int count;
    private double fps;
    synchronized void add(long timeNs) {
        if (timeNs<=0 || timeNs<=last) return;
        lastArrival=System.nanoTime();
        last=timeNs;
        if (first==0) { first=timeNs; count=1; return; }
        count++;
        if (timeNs-first>=1_000_000_000L) {
            fps=(count-1)*1e9/(timeNs-first);
            first=timeNs; count=1;
        }
    }
    synchronized double fps() { return System.nanoTime()-lastArrival>1_500_000_000L ? 0 : fps; }
}
