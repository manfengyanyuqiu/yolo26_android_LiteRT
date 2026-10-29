package com.fydnn.dimensity;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.view.View;
import java.util.List;
import java.util.Locale;

/** Boxes use the same rotated/letterboxed model canvas as the GPU preview. */
final class DetectionOverlay extends View {
    private record Frame(List<YoloDecoder.Box> boxes,String[] labels,int width,int height,long completedNs) {}
    private volatile Frame frame;
    private final Paint paint=new Paint(Paint.ANTI_ALIAS_FLAG);
    private final float density;
    DetectionOverlay(Context context) { super(context); density=getResources().getDisplayMetrics().density; }
    void update(List<YoloDecoder.Box> boxes,int width,int height) {
        String[] labels=new String[boxes.size()];
        for(int i=0;i<labels.length;i++) {var box=boxes.get(i);labels[i]=box.cls()+String.format(Locale.US," %.2f",box.score());}
        frame=new Frame(boxes,labels,width,height,System.nanoTime()); postInvalidateOnAnimation();
    }
    void clear() { frame=null; postInvalidateOnAnimation(); }
    @Override protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        Frame current=frame;
        if (current==null || System.nanoTime()-current.completedNs>500_000_000L) return;
        float sx=(float)getWidth()/current.width, sy=(float)getHeight()/current.height;
        paint.setColor(Color.GREEN); paint.setStrokeWidth(2*density); paint.setTextSize(13*density);
        for (int i=0;i<current.boxes.size();i++) {
            YoloDecoder.Box box=current.boxes.get(i);
            paint.setStyle(Paint.Style.STROKE);
            canvas.drawRect(box.left()*sx,box.top()*sy,box.right()*sx,box.bottom()*sy,paint);
            paint.setStyle(Paint.Style.FILL);
            canvas.drawText(current.labels[i],
                    box.left()*sx,Math.max(15*density,box.top()*sy-3*density),paint);
        }
    }
}
