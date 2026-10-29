package com.fydnn.dimensity;

import android.content.Context;
import android.graphics.SurfaceTexture;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.os.Handler;
import android.os.Looper;
import android.view.Surface;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

/** External-OES camera preview + model-sized GPU letterbox, without full-resolution CPU conversion. */
final class CameraGlView extends GLSurfaceView {
    interface Listener extends Camera2Source.Listener {
        boolean acquireFrame();
        void releaseFrame();
        void onPreviewFrame(long timestampNs);
        void onFrame(ByteBuffer rgba, long readbackStartNs, double gpuReadbackMs);
    }
    private final CameraRenderer renderer;
    CameraGlView(Context context,int width,int height,int displayRotation,Listener listener) {
        super(context);
        setEGLContextClientVersion(2);
        setEGLConfigChooser(8,8,8,8,0,0);
        setPreserveEGLContextOnPause(false);
        renderer=new CameraRenderer(context,width,height,displayRotation,listener);
        setRenderer(renderer); setRenderMode(RENDERMODE_WHEN_DIRTY);
    }
    void stopCamera() { renderer.stop(); }
    void releaseGl() { queueEvent(renderer::release); }

    private final class CameraRenderer implements Renderer {
        private final Context context;
        private final int modelWidth,modelHeight,displayRotation;
        private final Listener listener;
        private final float[] textureMatrix=new float[16];
        private final FloatBuffer vertices=floats(new float[8]);
        private final FloatBuffer textureCoordinates=floats(new float[]{0,0,1,0,0,1,1,1});
        private final ByteBuffer rgba;
        private volatile CameraModeSelector.Mode mode;
        private volatile Camera2Source camera;
        private volatile boolean stopped;
        private SurfaceTexture texture;
        private int cameraTexture,framebuffer,colorTexture,program;
        private int positionLocation,texCoordLocation,matrixLocation,samplerLocation;
        private int viewWidth,viewHeight;
        private long lastTimestamp;

        CameraRenderer(Context context,int width,int height,int displayRotation,Listener listener) {
            this.context=context; modelWidth=width; modelHeight=height;
            this.displayRotation=displayRotation; this.listener=listener;
            rgba=ByteBuffer.allocateDirect(width*height*4).order(ByteOrder.nativeOrder());
        }
        @Override public void onSurfaceCreated(GL10 ignored,EGLConfig config) {
            if (stopped) return;
            try {
                program=createProgram();
                positionLocation=GLES20.glGetAttribLocation(program,"aPosition");
                texCoordLocation=GLES20.glGetAttribLocation(program,"aTexCoord");
                matrixLocation=GLES20.glGetUniformLocation(program,"uTextureMatrix");
                samplerLocation=GLES20.glGetUniformLocation(program,"uCamera");
                int[] ids=new int[1]; GLES20.glGenTextures(1,ids,0); cameraTexture=ids[0];
                GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,cameraTexture);
                textureParameters(GLES11Ext.GL_TEXTURE_EXTERNAL_OES);
                texture=new SurfaceTexture(cameraTexture);
                texture.setOnFrameAvailableListener(t -> { if (!stopped) requestRender(); },new Handler(Looper.getMainLooper()));
                createFramebuffer();
                camera=new Camera2Source(context,new Camera2Source.Listener() {
                    @Override public void onMode(CameraModeSelector.Mode selected) { mode=selected; listener.onMode(selected); }
                    @Override public void onCapture(long timestamp) { listener.onCapture(timestamp); }
                    @Override public void onError(String message) { if (!stopped) listener.onError(message); }
                });
                if (stopped) camera.close(); else camera.start(texture);
            } catch (RuntimeException e) { error("摄像头 OpenGL 初始化失败："+e); }
        }
        @Override public void onSurfaceChanged(GL10 ignored,int width,int height) {
            viewWidth=width; viewHeight=height;
            Surface surface=getHolder().getSurface();
            if (surface.isValid()) surface.setFrameRate(120f,Surface.FRAME_RATE_COMPATIBILITY_FIXED_SOURCE);
        }
        @Override public void onDrawFrame(GL10 ignored) {
            if (stopped || texture == null || mode == null || viewWidth==0 || viewHeight==0) return;
            boolean acquired=false;
            try {
                texture.updateTexImage();
                long timestamp=texture.getTimestamp();
                if (timestamp==0 || timestamp==lastTimestamp) return;
                lastTimestamp=timestamp;
                texture.getTransformMatrix(textureMatrix);
                CameraModeSelector.Mode selected=mode;
                int rotation=(selected.sensorRotation()-displayRotation+360)%360;
                float[] positions=CameraGeometry.vertices(selected.width(),selected.height(),rotation,modelWidth,modelHeight);
                vertices.position(0); vertices.put(positions); vertices.position(0);
                listener.onPreviewFrame(timestamp);
                GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER,0);
                draw(viewWidth,viewHeight);
                if (listener.acquireFrame()) {
                    acquired=true;
                    long start=System.nanoTime();
                    GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER,framebuffer);
                    draw(modelWidth,modelHeight);
                    rgba.clear();
                    GLES20.glReadPixels(0,0,modelWidth,modelHeight,GLES20.GL_RGBA,GLES20.GL_UNSIGNED_BYTE,rgba);
                    checkGl("camera model readback");
                    rgba.position(0);
                    GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER,0);
                    GLES20.glViewport(0,0,viewWidth,viewHeight);
                    listener.onFrame(rgba,start,(System.nanoTime()-start)/1e6);
                    acquired=false; // Worker now owns this buffer until releaseFrame().
                }
            } catch (RuntimeException e) {
                if (acquired) listener.releaseFrame();
                error("摄像头帧处理失败："+e);
            }
        }
        private void draw(int width,int height) {
            GLES20.glViewport(0,0,width,height);
            GLES20.glClearColor(114/255f,114/255f,114/255f,1);
            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);
            GLES20.glUseProgram(program);
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,cameraTexture);
            GLES20.glUniform1i(samplerLocation,0);
            GLES20.glUniformMatrix4fv(matrixLocation,1,false,textureMatrix,0);
            GLES20.glEnableVertexAttribArray(positionLocation);
            GLES20.glEnableVertexAttribArray(texCoordLocation);
            vertices.position(0); textureCoordinates.position(0);
            GLES20.glVertexAttribPointer(positionLocation,2,GLES20.GL_FLOAT,false,0,vertices);
            GLES20.glVertexAttribPointer(texCoordLocation,2,GLES20.GL_FLOAT,false,0,textureCoordinates);
            GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP,0,4);
        }
        private void createFramebuffer() {
            int[] ids=new int[1]; GLES20.glGenTextures(1,ids,0); colorTexture=ids[0];
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D,colorTexture);
            textureParameters(GLES20.GL_TEXTURE_2D);
            GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D,0,GLES20.GL_RGBA,modelWidth,modelHeight,0,
                    GLES20.GL_RGBA,GLES20.GL_UNSIGNED_BYTE,null);
            GLES20.glGenFramebuffers(1,ids,0); framebuffer=ids[0];
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER,framebuffer);
            GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER,GLES20.GL_COLOR_ATTACHMENT0,
                    GLES20.GL_TEXTURE_2D,colorTexture,0);
            if (GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER)!=GLES20.GL_FRAMEBUFFER_COMPLETE)
                throw new IllegalStateException("Model-sized framebuffer incomplete");
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER,0);
            checkGl("create framebuffer");
        }
        private void error(String message) {
            if (!stopped) { listener.onError(message); stop(); }
        }
        void stop() { stopped=true; Camera2Source source=camera; if (source!=null) source.close(); }
        void release() {
            stop();
            if (texture!=null) { texture.setOnFrameAvailableListener(null); texture.release(); texture=null; }
            if (framebuffer!=0) GLES20.glDeleteFramebuffers(1,new int[]{framebuffer},0);
            if (colorTexture!=0) GLES20.glDeleteTextures(1,new int[]{colorTexture},0);
            if (cameraTexture!=0) GLES20.glDeleteTextures(1,new int[]{cameraTexture},0);
            if (program!=0) GLES20.glDeleteProgram(program);
            framebuffer=0; colorTexture=0; cameraTexture=0; program=0;
        }
    }
    private static FloatBuffer floats(float[] values) {
        FloatBuffer buffer=ByteBuffer.allocateDirect(values.length*4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        buffer.put(values).position(0); return buffer;
    }
    private static void textureParameters(int target) {
        GLES20.glTexParameteri(target,GLES20.GL_TEXTURE_MIN_FILTER,GLES20.GL_LINEAR);
        GLES20.glTexParameteri(target,GLES20.GL_TEXTURE_MAG_FILTER,GLES20.GL_LINEAR);
        GLES20.glTexParameteri(target,GLES20.GL_TEXTURE_WRAP_S,GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(target,GLES20.GL_TEXTURE_WRAP_T,GLES20.GL_CLAMP_TO_EDGE);
    }
    private static int shader(int kind,String source) {
        int shader=GLES20.glCreateShader(kind); GLES20.glShaderSource(shader,source); GLES20.glCompileShader(shader);
        int[] status=new int[1]; GLES20.glGetShaderiv(shader,GLES20.GL_COMPILE_STATUS,status,0);
        if (status[0]==0) {
            String message=GLES20.glGetShaderInfoLog(shader); GLES20.glDeleteShader(shader);
            throw new IllegalStateException("Shader: "+message);
        }
        return shader;
    }
    private static int createProgram() {
        int vertex=shader(GLES20.GL_VERTEX_SHADER,
                "attribute vec4 aPosition; attribute vec4 aTexCoord; uniform mat4 uTextureMatrix; varying vec2 vTexCoord;"
                +"void main(){gl_Position=aPosition;vTexCoord=(uTextureMatrix*aTexCoord).xy;}");
        int fragment=shader(GLES20.GL_FRAGMENT_SHADER,
                "#extension GL_OES_EGL_image_external : require\nprecision mediump float;"
                +"uniform samplerExternalOES uCamera;varying vec2 vTexCoord;"
                +"void main(){gl_FragColor=texture2D(uCamera,vTexCoord);}");
        int program=GLES20.glCreateProgram(); GLES20.glAttachShader(program,vertex); GLES20.glAttachShader(program,fragment);
        GLES20.glLinkProgram(program); GLES20.glDeleteShader(vertex); GLES20.glDeleteShader(fragment);
        int[] status=new int[1]; GLES20.glGetProgramiv(program,GLES20.GL_LINK_STATUS,status,0);
        if (status[0]==0) {
            String message=GLES20.glGetProgramInfoLog(program); GLES20.glDeleteProgram(program);
            throw new IllegalStateException("GL program: "+message);
        }
        return program;
    }
    private static void checkGl(String stage) {
        int error=GLES20.glGetError();
        if (error!=GLES20.GL_NO_ERROR) throw new IllegalStateException(stage+": GL error "+error);
    }
}
