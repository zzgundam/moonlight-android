package com.limelight.utils;

import android.content.Context;
import android.graphics.SurfaceTexture;
import android.opengl.GLES20;
import android.opengl.GLES30;
import android.opengl.GLSurfaceView;
import android.view.Surface;

import com.limelight.LimeLog;
import com.limelight.preferences.PreferenceConfiguration;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.concurrent.atomic.AtomicBoolean;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

/**
 * Real-time bidirectional frame interpolation for the 2D stream view.
 *
 * Pipeline: MediaCodec decodes into a SurfaceTexture; every incoming frame is
 * copied into a GL RGBA texture (prev/curr pair), a motion-vector field is
 * estimated once per real frame, and synthesized in-between frames are
 * presented on the display vsync grid at selectable 2x/3x/4x multipliers.
 *
 * Engine selection at surface creation:
 *   1. QCOM_ME       — Adreno hardware motion estimator (glTexEstimateMotionQCOM,
 *                      via JNI shim) feeding the bidirectional warp shader.
 *   2. QCOM_EXTRAP   — Adreno driver black-box extrapolator
 *                      (glExtrapolateTex2DQCOM) when motion estimation is
 *                      unavailable but extrapolation is.
 *   3. GLSL          — portable fallback: block-matching motion estimation
 *                      plus refinement, both in fragment shaders.
 *
 * MV conventions (all engines): mv = pos_curr - pos_prev, in pixels.
 * A synthesized frame at phase t in (0,1) samples prev at (p - mv*t) and
 * curr at (p + mv*(1-t)) and blends by t.
 */
public class FrameInterpolationRenderer implements GLSurfaceView.Renderer, SurfaceTexture.OnFrameAvailableListener {

    public interface OnSurfaceReadyListener {
        void onFrameInterpSurfaceReady(Surface surface);
    }

    private static final int GL_TEXTURE_EXTERNAL_OES = 0x8D65;
    private static final float[] QUAD_VERTICES = {-1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};
    // Standard (unflipped) GL texture coords: the SurfaceTexture transform
    // matrix handles buffer orientation, so no manual flip here.
    private static final float[] TEXTURE_VERTICES = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};

    // GLSL fallback motion estimation parameters: 8px estimation grid,
    // +/-6px coarse search (3px step) refined to 1px, 16 sparse luma taps
    // per candidate inside a 16x16 block. MV encoded in RGBA8 around 0.5
    // with +/-6px dynamic range.
    private static final int MV_GRID = 8;
    private static final float MV_RANGE = 6.0f;

    private static final long MIN_FRAME_INTERVAL_NS = 1_000_000L;
    private static final long MAX_FRAME_INTERVAL_NS = 100_000_000L;

    public static final int ENGINE_NONE = 0;
    public static final int ENGINE_QCOM_ME = 1;
    public static final int ENGINE_QCOM_EXTRAP = 2;
    public static final int ENGINE_GLSL = 3;

    // Public status fields (readable from overlays/debug UI)
    public static volatile int engine = ENGINE_NONE;
    public static volatile int multiplier = 2;
    public static volatile float displayFps = 0;
    public static volatile float sourceFps = 0;

    private final GLSurfaceView glSurfaceView;
    private final OnSurfaceReadyListener onSurfaceReadyListener;
    private final PreferenceConfiguration prefConfig;

    private final FloatBuffer quadVertexBuffer;
    private final FloatBuffer textureVertexBuffer;

    private final Object frameLock = new Object();
    private final AtomicBoolean frameAvailable = new AtomicBoolean(false);

    private SurfaceTexture videoSurfaceTexture;
    private Surface videoSurface;
    private int videoTextureId;

    // Programs
    private int copyProgram;        // EXT_OES -> RGBA FBO
    private int lumaProgram;        // RGBA -> R8 luma
    private int meProgram;          // coarse block matching -> encoded MV
    private int refineProgram;      // refine around coarse MV
    private int synthProgram;       // bidirectional warp + blend
    private int presentProgram;     // plain texture -> screen

    // Frame textures (swap pair)
    private int frameTexA, frameTexB;   // prev / curr color
    private int lumaTexA, lumaTexB;     // prev / curr luma (GL_R8)
    private int mvTexA, mvTexB;         // motion field ping/pong (GLSL path)
    private int mvTexHw;                // motion field (QCOM path, RGBA16F)
    private int frameFbo, lumaFbo, mvFboA, mvFboB;

    private int frameWidth = 0, frameHeight = 0;
    private final float[] stMatrix = new float[16];
    private boolean hasPrevFrame = false;
    private boolean glReady = false;
    private int qcomFeatures = 0;
    private int mvBlockX = 16, mvBlockY = 16;

    // Cadence state
    private long lastFrameNanos = 0;
    private double frameIntervalNanos = 16_666_666.0;
    private int slotsPresented = 0;
    private long fpsWindowStart = 0;
    private int displayFrameCount = 0;
    private int sourceFrameCount = 0;

    private boolean meFailed = false;

    static {
        try {
            System.loadLibrary("frame-interp");
            LimeLog.info("frame-interp native library loaded");
        } catch (UnsatisfiedLinkError e) {
            LimeLog.warning("frame-interp native library unavailable: " + e.getMessage());
        }
    }

    private static native int nativeQueryQcomFeatures();
    private static native boolean nativeEstimateMotion(int prevLuma, int currLuma, int mvTex);
    private static native boolean nativeExtrapolate(int prevTex, int currTex, int outTex, float factor);
    private static native boolean nativeQueryMotionBlock(BlockParams out);

    public static class BlockParams {
        public int x = 16;
        public int y = 16;
    }

    public FrameInterpolationRenderer(GLSurfaceView view, OnSurfaceReadyListener listener,
                                      Context context, PreferenceConfiguration prefConfig) {
        this.glSurfaceView = view;
        this.onSurfaceReadyListener = listener;
        this.prefConfig = prefConfig;
        FrameInterpolationRenderer.multiplier = Math.max(2, Math.min(4, prefConfig.frameInterpolationMultiplier));

        quadVertexBuffer = ByteBuffer.allocateDirect(QUAD_VERTICES.length * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        quadVertexBuffer.put(QUAD_VERTICES).position(0);
        textureVertexBuffer = ByteBuffer.allocateDirect(TEXTURE_VERTICES.length * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        textureVertexBuffer.put(TEXTURE_VERTICES).position(0);
    }

    public Surface getVideoSurface() {
        return videoSurface;
    }

    @Override
    public void onFrameAvailable(SurfaceTexture surfaceTexture) {
        synchronized (frameLock) {
            frameAvailable.set(true);
        }
        glSurfaceView.requestRender();
    }

    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        videoTextureId = createExternalOESTexture();
        videoSurfaceTexture = new SurfaceTexture(videoTextureId);
        videoSurfaceTexture.setOnFrameAvailableListener(this);
        videoSurface = new Surface(videoSurfaceTexture);

        // Detect QCOM hardware features with the current context active
        try {
            qcomFeatures = nativeQueryQcomFeatures();
        } catch (Throwable t) {
            LimeLog.warning("QCOM feature query failed: " + t.getMessage());
            qcomFeatures = 0;
        }
        LimeLog.info("Frame interpolation: QCOM features=0x" + Integer.toHexString(qcomFeatures));

        engine = chooseEngine();

        copyProgram = createProgram(VERT_SHADER, COPY_FRAG);
        lumaProgram = createProgram(VERT_SHADER, LUMA_FRAG);
        presentProgram = createProgram(VERT_SHADER, PRESENT_FRAG);

        if (engine == ENGINE_GLSL) {
            meProgram = createProgram(VERT_SHADER, ME_FRAG);
            refineProgram = createProgram(VERT_SHADER, ME_REFINE_FRAG);
            if (meProgram == 0 || refineProgram == 0) {
                LimeLog.severe("Frame interpolation: GLSL ME program failed to compile, interpolation disabled");
                engine = ENGINE_NONE;
            }
        }

        synthProgram = (engine == ENGINE_QCOM_EXTRAP)
                ? presentProgram
                : createProgram(VERT_SHADER, SYNTH_FRAG);

        if (engine != ENGINE_NONE && synthProgram == 0) {
            LimeLog.severe("Frame interpolation: synthesis program failed to compile, interpolation disabled");
            engine = ENGINE_NONE;
        }

        glReady = engine != ENGINE_NONE;

        if (engine != ENGINE_NONE && copyProgram == 0) {
            // Without the OES copy pass nothing can be displayed correctly.
            LimeLog.severe("Frame interpolation: OES copy program failed to compile, interpolation disabled");
            engine = ENGINE_NONE;
            glReady = false;
        }

        if (onSurfaceReadyListener != null) {
            onSurfaceReadyListener.onFrameInterpSurfaceReady(videoSurface);
        }
    }

    private int chooseEngine() {
        if ((qcomFeatures & 1) != 0) {
            return ENGINE_QCOM_ME;
        }
        if ((qcomFeatures & 2) != 0) {
            return ENGINE_QCOM_EXTRAP;
        }
        return ENGINE_GLSL;
    }

    @Override
    public void onSurfaceChanged(GL10 gl, int width, int height) {
        GLES20.glViewport(0, 0, width, height);
    }

    @Override
    public void onDrawFrame(GL10 gl) {
        long now = System.nanoTime();

        boolean newFrame;
        synchronized (frameLock) {
            newFrame = frameAvailable.getAndSet(false);
        }

        if (newFrame && glReady) {
            ingestFrame(now);
            presentReal();
            slotsPresented = 0;
            displayFrameCount++;
            sourceFrameCount++;
        } else {
            double progress = (now - lastFrameNanos) / frameIntervalNanos;
            if (!glReady || !hasPrevFrame || lastFrameNanos == 0) {
                if (glReady) {
                    presentReal();
                    displayFrameCount++;
                }
                return;
            }
            if (progress > 1.35) {
                // Stream stalled: hold the last real frame, no synthesis.
                presentReal();
                displayFrameCount++;
                return;
            }
            int maxSlots = multiplier - 1;
            int targetSlots = Math.min(maxSlots, (int) Math.floor(progress * multiplier));
            if (targetSlots > slotsPresented) {
                int slot = slotsPresented + 1;
                float phase = (float) slot / multiplier;
                presentSynthetic(phase, now);
                slotsPresented = slot;
                displayFrameCount++;
            }
        }

        updateFps(now);
    }

    private void ingestFrame(long now) {
        try {
            videoSurfaceTexture.updateTexImage();
            videoSurfaceTexture.getTransformMatrix(stMatrix);
        } catch (Exception e) {
            LimeLog.warning("Frame interpolation: updateTexImage failed: " + e.getMessage());
            return;
        }

        if (frameWidth == 0) {
            // The decoded stream size matches the configured stream resolution
            int w = prefConfig.width;
            int h = prefConfig.height;
            if (w <= 0 || h <= 0) {
                w = glSurfaceView.getWidth();
                h = glSurfaceView.getHeight();
            }
            if (w <= 0 || h <= 0) {
                return;
            }
            frameWidth = w;
            frameHeight = h;
            allocateFrameResources();
            LimeLog.info("Frame interpolation: frame size " + w + "x" + h
                    + ", engine=" + engineString() + ", multiplier=" + multiplier);
        }

        if (!glReady) {
            return;
        }

        // Copy decoded frame into the curr texture and its luma
        copyExternalToTexture(videoTextureId, frameTexB, stMatrix);
        renderLuma(frameTexB, lumaTexB);

        if (hasPrevFrame) {
            runMotionEstimation();
        }

        // Swap prev/curr
        int t = frameTexA; frameTexA = frameTexB; frameTexB = t;
        t = lumaTexA; lumaTexA = lumaTexB; lumaTexB = t;
        hasPrevFrame = true;

        if (lastFrameNanos != 0) {
            double delta = now - lastFrameNanos;
            if (delta >= MIN_FRAME_INTERVAL_NS && delta <= MAX_FRAME_INTERVAL_NS) {
                frameIntervalNanos = 0.8 * frameIntervalNanos + 0.2 * delta;
            }
        }
        lastFrameNanos = now;
    }

    private void runMotionEstimation() {
        if (engine == ENGINE_QCOM_ME) {
            boolean ok = false;
            try {
                ok = nativeEstimateMotion(lumaTexA, lumaTexB, mvTexHw);
            } catch (Throwable t) {
                LimeLog.warning("QCOM motion estimation crashed: " + t.getMessage());
            }
            if (!ok) {
                meFailed = true;
            }
        } else if (engine == ENGINE_GLSL) {
            // Coarse pass: prevMV (mvTexA) untouched this frame -> render into mvTexA
            renderMePass(meProgram, mvFboA, null, frameWidth, frameHeight);
            // Refine pass: read mvTexA, write mvTexB
            renderMePass(refineProgram, mvFboB, mvTexA, frameWidth, frameHeight);
        }
        // QCOM_EXTRAP synthesizes per-slot in presentSynthetic(); nothing here.
    }

    private void presentReal() {
        if (!glReady) {
            return;
        }
        // After the swap in ingestFrame(), frameTexA holds the newest frame.
        drawTextureToScreen(frameTexA != 0 ? frameTexA : frameTexB);
    }

    private void presentSynthetic(float phase, long now) {
        if (!glReady || !hasPrevFrame) {
            return;
        }
        if (engine == ENGINE_QCOM_EXTRAP) {
            // glExtrapolateTex2DQCOM expects (prev, curr, out, factor);
            // after the swap frameTexB is prev and frameTexA is curr.
            boolean ok = false;
            try {
                ok = nativeExtrapolate(frameTexB, frameTexA, getSynthTex(), phase);
            } catch (Throwable t) {
                LimeLog.warning("QCOM extrapolation crashed: " + t.getMessage());
            }
            if (ok) {
                drawTextureToScreen(synthTex);
            } else {
                drawTextureToScreen(frameTexA);
            }
            return;
        }

        int mvTex;
        float mvMode;
        if (engine == ENGINE_QCOM_ME) {
            mvTex = mvTexHw;
            mvMode = 1.0f; // raw pixels in RGBA16F
        } else {
            mvTex = mvTexB; // refined field, encoded
            mvMode = 0.0f; // encoded around 0.5 with MV_RANGE
        }

        GLES20.glUseProgram(synthProgram);
        bindQuad(synthProgram);

        // After the swap frameTexA = current frame, frameTexB = previous frame.
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, frameTexA); // curr
        GLES20.glUniform1i(GLES20.glGetUniformLocation(synthProgram, "uCurr"), 0);

        GLES20.glActiveTexture(GLES20.GL_TEXTURE1);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, frameTexB); // prev
        GLES20.glUniform1i(GLES20.glGetUniformLocation(synthProgram, "uPrev"), 1);

        GLES20.glActiveTexture(GLES20.GL_TEXTURE2);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, mvTex);
        GLES20.glUniform1i(GLES20.glGetUniformLocation(synthProgram, "uMV"), 2);

        GLES20.glUniform1f(GLES20.glGetUniformLocation(synthProgram, "uPhase"), phase);
        GLES20.glUniform1f(GLES20.glGetUniformLocation(synthProgram, "uMvMode"), mvMode);
        GLES20.glUniform1f(GLES20.glGetUniformLocation(synthProgram, "uMvRange"), MV_RANGE * 2.0f);
        GLES20.glUniform2f(GLES20.glGetUniformLocation(synthProgram, "uFrameSize"), frameWidth, frameHeight);
        GLES20.glUniform1f(GLES20.glGetUniformLocation(synthProgram, "uMaxMV"), MV_RANGE * 4.0f);

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);
    }

    private int synthTex = 0;

    private int getSynthTex() {
        if (synthTex == 0) {
            synthTex = createColorTexture(frameWidth, frameHeight);
        }
        return synthTex;
    }

    private void drawTextureToScreen(int tex) {
        GLES20.glUseProgram(presentProgram);
        bindQuad(presentProgram);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, tex);
        GLES20.glUniform1i(GLES20.glGetUniformLocation(presentProgram, "s_InputTexture"), 0);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);
    }

    private void bindQuad(int program) {
        int posHandle = GLES20.glGetAttribLocation(program, "a_Position");
        int texHandle = GLES20.glGetAttribLocation(program, "a_TexCoord");
        GLES20.glVertexAttribPointer(posHandle, 2, GLES20.GL_FLOAT, false, 0, quadVertexBuffer);
        GLES20.glVertexAttribPointer(texHandle, 2, GLES20.GL_FLOAT, false, 0, textureVertexBuffer);
        GLES20.glEnableVertexAttribArray(posHandle);
        GLES20.glEnableVertexAttribArray(texHandle);
    }

    private void copyExternalToTexture(int extTex, int destTex, float[] transform) {
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, frameFbo);
        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0, GLES20.GL_TEXTURE_2D, destTex, 0);
        GLES20.glViewport(0, 0, frameWidth, frameHeight);
        GLES20.glUseProgram(copyProgram);
        bindQuad(copyProgram);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GL_TEXTURE_EXTERNAL_OES, extTex);
        GLES20.glUniform1i(GLES20.glGetUniformLocation(copyProgram, "s_InputTexture"), 0);
        GLES20.glUniformMatrix4fv(GLES20.glGetUniformLocation(copyProgram, "uTransform"), 1, false, transform, 0);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
        GLES20.glBindTexture(GL_TEXTURE_EXTERNAL_OES, 0);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
    }

    private void renderLuma(int srcTex, int destTex) {
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, lumaFbo);
        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0, GLES20.GL_TEXTURE_2D, destTex, 0);
        GLES20.glViewport(0, 0, frameWidth, frameHeight);
        GLES20.glUseProgram(lumaProgram);
        bindQuad(lumaProgram);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, srcTex);
        GLES20.glUniform1i(GLES20.glGetUniformLocation(lumaProgram, "s_InputTexture"), 0);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
    }

    private void renderMePass(int program, int fbo, Integer inputMvTex, int w, int h) {
        int mvW = (w + MV_GRID - 1) / MV_GRID;
        int mvH = (h + MV_GRID - 1) / MV_GRID;
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fbo);
        GLES20.glViewport(0, 0, mvW, mvH);
        GLES20.glUseProgram(program);
        bindQuad(program);

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, lumaTexB);
        GLES20.glUniform1i(GLES20.glGetUniformLocation(program, "uCurrLuma"), 0);

        GLES20.glActiveTexture(GLES20.GL_TEXTURE1);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, lumaTexA);
        GLES20.glUniform1i(GLES20.glGetUniformLocation(program, "uPrevLuma"), 1);

        if (inputMvTex != null) {
            GLES20.glActiveTexture(GLES20.GL_TEXTURE2);
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, inputMvTex);
            GLES20.glUniform1i(GLES20.glGetUniformLocation(program, "uCoarseMV"), 2);
        }

        GLES20.glUniform1f(GLES20.glGetUniformLocation(program, "uMvRange"), MV_RANGE);
        GLES20.glUniform1f(GLES20.glGetUniformLocation(program, "uMvGrid"), MV_GRID);
        GLES20.glUniform2f(GLES20.glGetUniformLocation(program, "uFrameSize"), w, h);

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
    }

    private void allocateFrameResources() {
        frameTexA = createColorTexture(frameWidth, frameHeight);
        frameTexB = createColorTexture(frameWidth, frameHeight);
        lumaTexA = createR8Texture(frameWidth, frameHeight);
        lumaTexB = createR8Texture(frameWidth, frameHeight);

        int[] fbos = new int[1];
        GLES20.glGenFramebuffers(1, fbos, 0);
        frameFbo = fbos[0];
        GLES20.glGenFramebuffers(1, fbos, 0);
        lumaFbo = fbos[0];

        if (engine == ENGINE_GLSL) {
            int mvW = (frameWidth + MV_GRID - 1) / MV_GRID;
            int mvH = (frameHeight + MV_GRID - 1) / MV_GRID;
            mvTexA = createColorTexture(mvW, mvH);
            mvTexB = createColorTexture(mvW, mvH);
            GLES20.glGenFramebuffers(1, fbos, 0);
            mvFboA = fbos[0];
            GLES20.glGenFramebuffers(1, fbos, 0);
            mvFboB = fbos[0];
        } else if (engine == ENGINE_QCOM_ME) {
            BlockParams params = new BlockParams();
            boolean ok = false;
            try {
                ok = nativeQueryMotionBlock(params);
            } catch (Throwable t) {
                LimeLog.warning("QCOM block query failed: " + t.getMessage());
            }
            if (ok && params.x > 0 && params.y > 0) {
                mvBlockX = params.x;
                mvBlockY = params.y;
            }
            mvTexHw = createRgba16fTexture(
                    (frameWidth + mvBlockX - 1) / mvBlockX,
                    (frameHeight + mvBlockY - 1) / mvBlockY);
            LimeLog.info("Frame interpolation: QCOM ME block " + mvBlockX + "x" + mvBlockY);
        }
    }

    private void updateFps(long now) {
        if (fpsWindowStart == 0) {
            fpsWindowStart = now;
            return;
        }
        long elapsed = now - fpsWindowStart;
        if (elapsed >= 1_000_000_000L) {
            displayFps = displayFrameCount * 1_000_000_000f / elapsed;
            sourceFps = sourceFrameCount * 1_000_000_000f / elapsed;
            displayFrameCount = 0;
            sourceFrameCount = 0;
            fpsWindowStart = now;
            LimeLog.info("Frame interpolation: display=" + String.format("%.1f", displayFps)
                    + " source=" + String.format("%.1f", sourceFps)
                    + " engine=" + engineString() + " x" + multiplier);
        }
    }

    public String engineString() {
        switch (engine) {
            case ENGINE_QCOM_ME: return "QCOM-ME";
            case ENGINE_QCOM_EXTRAP: return "QCOM-EXTRAP";
            case ENGINE_GLSL: return "GLSL";
            default: return "OFF";
        }
    }

    public void onSurfaceDestroyed() {
        LimeLog.info("Shutting down frame interpolation renderer.");
        synchronized (frameLock) {
            frameAvailable.set(false);
        }
        if (videoSurface != null) {
            videoSurface.release();
            videoSurface = null;
        }
        if (videoSurfaceTexture != null) {
            videoSurfaceTexture.release();
            videoSurfaceTexture = null;
        }
        if (glSurfaceView != null) {
            glSurfaceView.queueEvent(() -> {
                int[] programs = {copyProgram, lumaProgram, meProgram, refineProgram, synthProgram, presentProgram};
                for (int p : programs) {
                    if (p != 0) {
                        GLES20.glDeleteProgram(p);
                    }
                }
                int[] textures = {videoTextureId, frameTexA, frameTexB, lumaTexA, lumaTexB,
                        mvTexA, mvTexB, mvTexHw, synthTex};
                int count = 0;
                int[] compact = new int[textures.length];
                for (int t : textures) {
                    if (t != 0) compact[count++] = t;
                }
                if (count > 0) {
                    GLES20.glDeleteTextures(count, compact, 0);
                }
                int[] fbos = {frameFbo, lumaFbo, mvFboA, mvFboB};
                int fCount = 0;
                int[] fCompact = new int[fbos.length];
                for (int f : fbos) {
                    if (f != 0) fCompact[fCount++] = f;
                }
                if (fCount > 0) {
                    GLES20.glDeleteFramebuffers(fCount, fCompact, 0);
                }
            });
        }
        glReady = false;
        hasPrevFrame = false;
        frameWidth = 0;
        frameHeight = 0;
        lastFrameNanos = 0;
        engine = ENGINE_NONE;
        displayFps = 0;
        sourceFps = 0;
    }

    // ─── GL helpers ──────────────────────────────────────────────────────────

    private int createExternalOESTexture() {
        int[] textures = new int[1];
        GLES20.glGenTextures(1, textures, 0);
        int textureId = textures[0];
        GLES20.glBindTexture(GL_TEXTURE_EXTERNAL_OES, textureId);
        GLES20.glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        return textureId;
    }

    private int createColorTexture(int width, int height) {
        int[] textures = new int[1];
        GLES20.glGenTextures(1, textures, 0);
        int textureId = textures[0];
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureId);
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, width, height, 0, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        return textureId;
    }

    private int createR8Texture(int width, int height) {
        int[] textures = new int[1];
        GLES30.glGenTextures(1, textures, 0);
        int textureId = textures[0];
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, textureId);
        GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_R8, width, height, 0, GLES30.GL_RED, GLES30.GL_UNSIGNED_BYTE, null);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR);
        return textureId;
    }

    private int createRgba16fTexture(int width, int height) {
        int[] textures = new int[1];
        GLES30.glGenTextures(1, textures, 0);
        int textureId = textures[0];
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, textureId);
        GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA16F, width, height, 0, GLES30.GL_RGBA, GLES30.GL_HALF_FLOAT, null);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR);
        return textureId;
    }

    private int loadShader(int type, String shaderCode) {
        int shader = GLES20.glCreateShader(type);
        GLES20.glShaderSource(shader, shaderCode);
        GLES20.glCompileShader(shader);
        int[] compiled = new int[1];
        GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, compiled, 0);
        if (compiled[0] == 0) {
            LimeLog.severe("Could not compile shader " + type + ":");
            LimeLog.severe(GLES20.glGetShaderInfoLog(shader));
            GLES20.glDeleteShader(shader);
            shader = 0;
        }
        return shader;
    }

    private int createProgram(String vertex, String fragment) {
        int vertexShader = loadShader(GLES20.GL_VERTEX_SHADER, vertex);
        if (vertexShader == 0) return 0;
        int fragmentShader = loadShader(GLES20.GL_FRAGMENT_SHADER, fragment);
        if (fragmentShader == 0) return 0;

        int program = GLES20.glCreateProgram();
        if (program != 0) {
            GLES20.glAttachShader(program, vertexShader);
            GLES20.glAttachShader(program, fragmentShader);
            GLES20.glLinkProgram(program);
            int[] linkStatus = new int[1];
            GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, linkStatus, 0);
            if (linkStatus[0] != GLES20.GL_TRUE) {
                LimeLog.severe("Could not link program: ");
                LimeLog.severe(GLES20.glGetProgramInfoLog(program));
                GLES20.glDeleteProgram(program);
                program = 0;
            }
        }
        return program;
    }

    // ─── Shaders ─────────────────────────────────────────────────────────────

    private static final String VERT_SHADER =
            "#version 300 es\n" +
            "precision mediump float;\n" +
            "in vec2 a_Position;\n" +
            "in vec2 a_TexCoord;\n" +
            "out vec2 vUV;\n" +
            "void main() {\n" +
            "    vUV = a_TexCoord;\n" +
            "    gl_Position = vec4(a_Position, 0.0, 1.0);\n" +
            "}\n";

    private static final String COPY_FRAG =
            "#version 300 es\n" +
            "#extension GL_OES_EGL_image_external_essl3 : require\n" +
            "precision mediump float;\n" +
            "uniform mediump samplerExternalOES s_InputTexture;\n" +
            "uniform highp mat4 uTransform;\n" +
            "in vec2 vUV;\n" +
            "out vec4 outColor;\n" +
            "void main() {\n" +
            "    vec2 tuv = (uTransform * vec4(vUV, 0.0, 1.0)).xy;\n" +
            "    outColor = texture(s_InputTexture, tuv);\n" +
            "}\n";

    private static final String LUMA_FRAG =
            "#version 300 es\n" +
            "precision mediump float;\n" +
            "uniform mediump sampler2D s_InputTexture;\n" +
            "in vec2 vUV;\n" +
            "out vec4 outColor;\n" +
            "void main() {\n" +
            "    vec3 c = texture(s_InputTexture, vUV).rgb;\n" +
            "    float l = 0.299 * c.r + 0.587 * c.g + 0.114 * c.b;\n" +
            "    outColor = vec4(l, 0.0, 0.0, 1.0);\n" +
            "}\n";

    private static final String PRESENT_FRAG =
            "#version 300 es\n" +
            "precision mediump float;\n" +
            "uniform mediump sampler2D s_InputTexture;\n" +
            "in vec2 vUV;\n" +
            "out vec4 outColor;\n" +
            "void main() {\n" +
            "    outColor = texture(s_InputTexture, vUV);\n" +
            "}\n";

    // Coarse motion estimation: 7x7 candidate grid, 3px step, +/-6px range,
    // 16 sparse taps in a 16x16 block. Output RGBA8:
    //   rg = mv encoded around 0.5 (mv/MV_RANGE/2 + 0.5)
    //   b  = confidence 0..1
    //   a  = valid flag
    private static final String ME_FRAG =
            "#version 300 es\n" +
            "precision highp float;\n" +
            "uniform mediump sampler2D uCurrLuma;\n" +
            "uniform mediump sampler2D uPrevLuma;\n" +
            "uniform float uMvRange;\n" +
            "uniform float uMvGrid;\n" +
            "uniform vec2 uFrameSize;\n" +
            "in vec2 vUV;\n" +
            "out vec4 outColor;\n" +
            "\n" +
            "float tapDiff(vec2 curPx, vec2 mv) {\n" +
            "    float sum = 0.0;\n" +
            "    for (int by = 0; by < 4; by++) {\n" +
            "        for (int bx = 0; bx < 4; bx++) {\n" +
            "            vec2 off = vec2(2.0 + 4.0 * float(bx), 2.0 + 4.0 * float(by));\n" +
            "            vec2 pc = curPx + off;\n" +
            "            vec2 pp = pc - mv;\n" +
            "            float lc = texture(uCurrLuma, pc / uFrameSize).r;\n" +
            "            float lp = texture(uPrevLuma, pp / uFrameSize).r;\n" +
            "            sum += abs(lc - lp);\n" +
            "        }\n" +
            "    }\n" +
            "    return sum;\n" +
            "}\n" +
            "\n" +
            "void main() {\n" +
            "    vec2 blockCenter = vUV * uFrameSize;\n" +
            "    float best = 1e9;\n" +
            "    vec2 bestMv = vec2(0.0);\n" +
            "    for (int dy = -2; dy <= 2; dy++) {\n" +
            "        for (int dx = -2; dx <= 2; dx++) {\n" +
            "            vec2 mv = vec2(float(dx), float(dy)) * (uMvRange * 0.5);\n" +
            "            float sad = tapDiff(blockCenter, mv);\n" +
            "            if (sad < best) {\n" +
            "                best = sad;\n" +
            "                bestMv = mv;\n" +
            "            }\n" +
            "        }\n" +
            "    }\n" +
            "    float conf = clamp(1.0 - best / (16.0 * 0.6), 0.0, 1.0);\n" +
            "    vec2 enc = bestMv / uMvRange * 0.5 + 0.5;\n" +
            "    outColor = vec4(enc, conf, 1.0);\n" +
            "}\n";

    // Refinement pass around the coarse result: +/-1px, 16 taps.
    private static final String ME_REFINE_FRAG =
            "#version 300 es\n" +
            "precision highp float;\n" +
            "uniform mediump sampler2D uCurrLuma;\n" +
            "uniform mediump sampler2D uPrevLuma;\n" +
            "uniform mediump sampler2D uCoarseMV;\n" +
            "uniform float uMvRange;\n" +
            "uniform vec2 uFrameSize;\n" +
            "in vec2 vUV;\n" +
            "out vec4 outColor;\n" +
            "\n" +
            "float tapDiff(vec2 curPx, vec2 mv) {\n" +
            "    float sum = 0.0;\n" +
            "    for (int by = 0; by < 4; by++) {\n" +
            "        for (int bx = 0; bx < 4; bx++) {\n" +
            "            vec2 off = vec2(2.0 + 4.0 * float(bx), 2.0 + 4.0 * float(by));\n" +
            "            vec2 pc = curPx + off;\n" +
            "            vec2 pp = pc - mv;\n" +
            "            float lc = texture(uCurrLuma, pc / uFrameSize).r;\n" +
            "            float lp = texture(uPrevLuma, pp / uFrameSize).r;\n" +
            "            sum += abs(lc - lp);\n" +
            "        }\n" +
            "    }\n" +
            "    return sum;\n" +
            "}\n" +
            "\n" +
            "void main() {\n" +
            "    vec2 blockCenter = vUV * uFrameSize;\n" +
            "    vec4 coarse = texture(uCoarseMV, vUV);\n" +
            // Encode convention (both ME passes): enc = mv / (2*uMvRange) + 0.5,
            // so decoding must multiply by 2*uMvRange to recover pixels.
            "    vec2 startMv = (coarse.rg - 0.5) * 2.0 * uMvRange;\n" +
            "    float best = 1e9;\n" +
            "    vec2 bestMv = startMv;\n" +
            "    for (int dy = -1; dy <= 1; dy++) {\n" +
            "        for (int dx = -1; dx <= 1; dx++) {\n" +
            "            vec2 mv = startMv + vec2(float(dx), float(dy));\n" +
            "            float sad = tapDiff(blockCenter, mv);\n" +
            "            if (sad < best) {\n" +
            "                best = sad;\n" +
            "                bestMv = mv;\n" +
            "            }\n" +
            "        }\n" +
            "    }\n" +
            "    float conf = clamp(1.0 - best / (16.0 * 0.6), 0.0, 1.0);\n" +
            "    vec2 enc = bestMv / uMvRange * 0.5 + 0.5;\n" +
            "    outColor = vec4(enc, conf, 1.0);\n" +
            "}\n";

    // Bidirectional synthesis. uPhase in (0,1): 0 = prev, 1 = curr.
    // uMvMode 0: encoded RGBA8 field (GLSL path); 1: raw pixels in RGBA16F (QCOM).
    private static final String SYNTH_FRAG =
            "#version 300 es\n" +
            "precision highp float;\n" +
            "uniform mediump sampler2D uCurr;\n" +
            "uniform mediump sampler2D uPrev;\n" +
            "uniform highp sampler2D uMV;\n" +
            "uniform float uPhase;\n" +
            "uniform float uMvMode;\n" +
            "uniform float uMvRange;\n" +
            "uniform vec2 uFrameSize;\n" +
            "uniform float uMaxMV;\n" +
            "in vec2 vUV;\n" +
            "out vec4 outColor;\n" +
            "\n" +
            "void main() {\n" +
            "    vec4 m = texture(uMV, vUV);\n" +
            "    vec2 mvPx = mix((m.rg - 0.5) * uMvRange, m.rg, uMvMode);\n" +
            "    float conf = clamp(m.b, 0.0, 1.0);\n" +
            "\n" +
            "    float len = length(mvPx);\n" +
            "    if (len > uMaxMV) {\n" +
            "        mvPx = vec2(0.0);\n" +
            "        conf = 0.0;\n" +
            "    }\n" +
            "\n" +
            "    vec2 mvUv = mvPx / uFrameSize;\n" +
            "    vec2 prevUv = vUV - mvUv * uPhase;\n" +
            "    vec2 currUv = vUV + mvUv * (1.0 - uPhase);\n" +
            "\n" +
            "    vec2 pc = clamp(prevUv, vec2(0.0), vec2(1.0));\n" +
            "    vec2 cc = clamp(currUv, vec2(0.0), vec2(1.0));\n" +
            "    float outside = (pc != prevUv || cc != currUv) ? 1.0 : 0.0;\n" +
            "\n" +
            "    vec4 cPrev = texture(uPrev, pc);\n" +
            "    vec4 cCurr = texture(uCurr, cc);\n" +
            "\n" +
            "    vec4 blended = mix(cPrev, cCurr, uPhase);\n" +
            "    vec4 fallback = mix(cCurr, blended, 0.5 * uPhase);\n" +
            "    outColor = mix(fallback, blended, conf);\n" +
            "    outColor = mix(outColor, cCurr, outside * (1.0 - uPhase * 0.5));\n" +
            "}\n";

    // ─── Imports that need no ambiguity ─────────────────────────────────────
    private static final String TAG = "FrameInterp";
}
