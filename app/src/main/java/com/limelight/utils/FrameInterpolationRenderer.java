package com.limelight.utils;

import android.content.Context;
import android.graphics.SurfaceTexture;
import android.opengl.GLES20;
import android.opengl.GLES30;
import android.opengl.GLSurfaceView;
import android.view.Surface;

import com.limelight.LimeLog;
import com.limelight.preferences.PreferenceConfiguration;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
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
 *   1. QCOM_ME — Adreno hardware motion estimator (glTexEstimateMotionQCOM,
 *                via JNI shim) feeding the bidirectional warp shader.
 *   2. GLSL    — portable fallback: block-matching motion estimation
 *                plus refinement, both in fragment shaders.
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

    // Post-compensation residual gate: when the motion-compensated prev and
    // curr samples disagree by more than this (average RGB difference, 0..1),
    // the motion is not trustworthy (scene cut, wrong MV, occlusion) and the
    // synthesis falls back to the nearest real frame instead of blending.
    // User-tunable ("interp confidence gate"): the original fixed 0.22 passed
    // video content wholesale — motion blur and codec noise sit at 5-15%
    // mismatch, so nearly every pixel blended and panning shots ghosted.
    private static final float CUT_THRESHOLD = 0.22f;
    private static final float[] GATE_THRESHOLDS = {0.22f, 0.15f, 0.12f, 0.08f};
    // Blur/softening levels (user setting): temporal persistence strength.
    // 0 disables the pass; higher values leave longer motion trails.
    private static final float[] BLUR_STRENGTHS = {0.0f, 0.3f, 0.5f, 0.7f};

    private static final long MIN_FRAME_INTERVAL_NS = 1_000_000L;
    private static final long MAX_FRAME_INTERVAL_NS = 100_000_000L;

    public static final int ENGINE_NONE = 0;
    public static final int ENGINE_QCOM_ME = 1;
    public static final int ENGINE_GLSL = 3;

    // Public status fields (readable from overlays/debug UI)
    public static volatile boolean isActive = false;
    public static volatile int engine = ENGINE_NONE;
    public static volatile int multiplier = 2;
    public static volatile float displayFps = 0;
    public static volatile float sourceFps = 0;

    private final GLSurfaceView glSurfaceView;
    private final OnSurfaceReadyListener onSurfaceReadyListener;
    private final PreferenceConfiguration prefConfig;
    private final boolean fbcEnabled;   // user setting: backward ME / FBC occlusion check
    private final float phaseAdvance;   // user setting: how much earlier synthetic frames claim a vsync slot
    private final double stallThresh;   // user setting: progress beyond which the stream is treated as stalled
    private final float cutThreshold;   // user setting: post-compensation residual gate (ghosting vs coverage)
    private final boolean interpEnabled; // false = pass-through: present decoded frames as-is, no ME/synthesis
    private final float blurStrength;   // user setting: temporal persistence strength (0 = off)

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
    private int lumaHalfProgram;    // RGBA -> half-res R8 luma (4-tap box)
    private int meProgram;          // coarse block matching -> encoded MV
    private int refineProgram;      // refine around coarse MV
    private int fuseProgram;        // multi-scale MV candidate fusion (QCOM)
    private int smoothProgram;      // 3x3 median MV smoothing (QCOM)
    private int synthProgram;       // bidirectional warp + blend
    private int presentProgram;     // plain texture -> screen
    private int inpaintProgram;     // fill invalid (occluded) pixels from valid neighbours

    // Frame textures (swap pair)
    private int frameTexA, frameTexB;   // prev / curr color
    private int lumaTexA, lumaTexB;     // prev / curr luma (GL_R8)
    private int lumaHalfA, lumaHalfB;   // prev / curr luma at 1/2 res (QCOM multi-scale)
    private int mvTexA, mvTexB;         // motion field ping/pong (GLSL path)
    private int mvTexHw;                // motion field (QCOM path, full res blocks)
    private int mvTexHwHalf;            // motion field (QCOM path, half-res input -> 2x range)
    private int mvTexBwd;               // backward motion field (QCOM FBC)
    private int mvBwdSm;                // smoothed backward field (QCOM FBC)
    private int mvRawFusedTex;          // fused candidate field (block grid)
    private int mvSmA, mvSmB;           // smoothed field ping/pong (smCurr / smPrev)
    private int smCurr, smPrev;
    private int frameFbo, lumaFbo, lumaHalfFbo, mvFboA, mvFboB, fuseFbo, smoothFbo;

    private int frameWidth = 0, frameHeight = 0;
    private int surfaceWidth = 0, surfaceHeight = 0;
    private int mvGridW = 0, mvGridH = 0;   // block grid of the active MV field
    private final float[] stMatrix = new float[16];
    private boolean hasPrevFrame = false;
    private boolean glReady = false;
    private int qcomFeatures = 0;
    private int mvBlockX = 16, mvBlockY = 16;
    private volatile boolean halfScaleMe = true;   // half-res HW ME usable
    private int halfScaleFailCount = 0;            // consecutive half-res ME failures
    private volatile boolean temporalPred = true;  // previous MV field usable
    private volatile boolean bwdFieldValid = false; // backward field (FBC) usable
    private float fbcThresh = 6.0f;                 // fwd+bwd disagreement cap (px)

    // Cadence state
    private long lastFrameNanos = 0;
    private double frameIntervalNanos = 16_666_666.0;
    private int slotsPresented = 0;
    private long fpsWindowStart = 0;
    private int displayFrameCount = 0;
    private int sourceFrameCount = 0;

    private boolean meFailed = false;

    private long ingestAccumNs = 0;
    private int ingestCount = 0;

    // Diagnostics: per-second cadence counters, mirrored to a file log so a
    // session can be analysed offline (why the multiplier is not reached).
    private int loopFrameCount = 0;      // every onDrawFrame callback
    private int realPresentCount = 0;    // real frames swapped to screen
    private int synthPresentCount = 0;   // synthetic frames swapped to screen
    private int repeatPresentCount = 0;  // vsync slots that re-show the last frame
    private int gateMissCount = 0;       // repeats where the phase gate was NOT reached
    private int stallHoldCount = 0;      // stream-stall holds (progress > 1.35)

    private Context logContext = null;
    private final boolean diagLogEnabled;
    private File diagFile = null;
    private BufferedWriter diagWriter = null;
    private boolean diagSessionHeader = false;
    private static final long DIAG_LOG_MAX_BYTES = 8L * 1024 * 1024;

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
        this.logContext = context;
        this.diagLogEnabled = prefConfig.frameInterpolationDiagLog;
        this.fbcEnabled = prefConfig.frameInterpolationFbc;
        int phaseLevel = Math.max(0, Math.min(3, prefConfig.frameInterpolationPhase));
        this.phaseAdvance = phaseLevel * 0.10f;      // 0.0 / 0.1 / 0.2 / 0.3
        this.stallThresh = 1.35 + phaseLevel * 0.15; // 1.35 / 1.50 / 1.65 / 1.80
        int gateLevel = Math.max(0, Math.min(3, prefConfig.frameInterpolationGate));
        this.cutThreshold = GATE_THRESHOLDS[gateLevel]; // 0.22 / 0.15 / 0.12 / 0.08
        this.interpEnabled = prefConfig.frameInterpolation;
        int blurLevel = Math.max(0, Math.min(3, prefConfig.frameInterpolationBlur));
        this.blurStrength = BLUR_STRENGTHS[blurLevel]; // 0.0 / 0.3 / 0.5 / 0.7
        FrameInterpolationRenderer.multiplier = Math.max(2, Math.min(4, prefConfig.frameInterpolationMultiplier));
        // Seed the cadence estimate from the configured stream FPS instead of
        // assuming 60; a 30 FPS source would otherwise look "stalled" for the
        // first ~10 frames and skip interpolation at startup (B5).
        this.frameIntervalNanos = defaultFrameIntervalNanos();

        quadVertexBuffer = ByteBuffer.allocateDirect(QUAD_VERTICES.length * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        quadVertexBuffer.put(QUAD_VERTICES).position(0);
        textureVertexBuffer = ByteBuffer.allocateDirect(TEXTURE_VERTICES.length * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        textureVertexBuffer.put(TEXTURE_VERTICES).position(0);
    }

    private double defaultFrameIntervalNanos() {
        return (prefConfig != null && prefConfig.fps > 0)
                ? 1_000_000_000.0 / prefConfig.fps
                : 16_666_666.0;
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
        halfScaleMe = true;
        halfScaleFailCount = 0;
        temporalPred = false;
        meFailed = false;
        bwdFieldValid = false;
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
        lumaHalfProgram = createProgram(VERT_SHADER, LUMA_HALF_FRAG);
        presentProgram = createProgram(VERT_SHADER, PRESENT_FRAG);

        if (engine == ENGINE_QCOM_ME) {
            fuseProgram = createProgram(VERT_SHADER, FUSE_FRAG);
            smoothProgram = createProgram(VERT_SHADER, SMOOTH_FRAG);
            if (fuseProgram == 0 || smoothProgram == 0) {
                // Multi-scale fusion unavailable: fall back to the raw
                // hardware field (still correct, just a smaller search range).
                LimeLog.warning("QCOM MV fusion programs unavailable, using raw HW field");
                halfScaleMe = false;
            }
        }

        if (engine == ENGINE_GLSL) {
            meProgram = createProgram(VERT_SHADER, ME_FRAG);
            refineProgram = createProgram(VERT_SHADER, ME_REFINE_FRAG);
            if (meProgram == 0 || refineProgram == 0) {
                LimeLog.severe("Frame interpolation: GLSL ME program failed to compile, interpolation disabled");
                engine = ENGINE_NONE;
            }
        }

        synthProgram = createProgram(VERT_SHADER, SYNTH_FRAG);

        inpaintProgram = createProgram(VERT_SHADER, INPAINT_FRAG);
        if (inpaintProgram == 0) {
            // Occlusion inpainting is an enhancement, not a requirement.
            LimeLog.warning("Occlusion inpaint program unavailable, using nearest-frame fallback");
        }

        metricsProgram = createProgram(VERT_SHADER, METRICS_FRAG);

        if (engine != ENGINE_NONE && synthProgram == 0) {
            LimeLog.severe("Frame interpolation: synthesis program failed to compile, interpolation disabled");
            engine = ENGINE_NONE;
        }

        glReady = engine != ENGINE_NONE;
        isActive = glReady;

        if (engine != ENGINE_NONE && copyProgram == 0) {
            // Without the OES copy pass nothing can be displayed correctly.
            LimeLog.severe("Frame interpolation: OES copy program failed to compile, interpolation disabled");
            engine = ENGINE_NONE;
            glReady = false;
            isActive = false;
        }

        if (onSurfaceReadyListener != null) {
            onSurfaceReadyListener.onFrameInterpSurfaceReady(videoSurface);
        }
    }

    private int chooseEngine() {
        if ((qcomFeatures & 1) != 0) {
            return ENGINE_QCOM_ME;
        }
        return ENGINE_GLSL;
    }

    @Override
    public void onSurfaceChanged(GL10 gl, int width, int height) {
        surfaceWidth = width;
        surfaceHeight = height;
        applyScreenViewport();
    }

    /**
     * Set the viewport for drawing to the screen (default framebuffer).
     * StreamContainer aspect-fits the GLSurfaceView to the video, so the
     * frame stretches to the full view exactly like the plain 2D SurfaceView
     * path (including fillDisplay/zoom). Every screen draw must call this —
     * FBO passes leave a different viewport behind, and unstretched regions
     * of the back buffer would show garbage.
     */
    private void applyScreenViewport() {
        int sw = surfaceWidth > 0 ? surfaceWidth : glSurfaceView.getWidth();
        int sh = surfaceHeight > 0 ? surfaceHeight : glSurfaceView.getHeight();
        if (sw <= 0 || sh <= 0) {
            return;
        }
        GLES20.glViewport(0, 0, sw, sh);
    }

    @Override
    public void onDrawFrame(GL10 gl) {
        long now = System.nanoTime();
        loopFrameCount++;

        boolean newFrame;
        synchronized (frameLock) {
            newFrame = frameAvailable.getAndSet(false);
        }

        if (!interpEnabled) {
            // Pass-through mode (interpolation disabled): present every
            // decoded frame at the next vsync and repeat the last presented
            // frame otherwise. No slots, no synthesis, no motion estimation.
            if (newFrame && glReady) {
                long t0 = System.nanoTime();
                ingestFrame(now);
                ingestAccumNs += System.nanoTime() - t0;
                ingestCount++;
                presentReal();
                displayFrameCount++;
                sourceFrameCount++;
            } else if (glReady && lastScreenTex != 0) {
                repeatPresentCount++;
                drawTextureToScreen(lastScreenTex);
            }
            updateFps(now);
            return;
        }

        if (newFrame && glReady) {
            // A newly decoded frame always takes this vsync. A GLSurfaceView
            // swaps once per onDrawFrame, so "catching up" several overdue
            // synthetic slots inside a single callback was pointless: only the
            // last draw reached the screen, while every pass counted as a
            // displayed frame and inflated displayFps (A6). In-between frames
            // are produced on the vsyncs BETWEEN real frames (else branch).
            long t0 = System.nanoTime();
            ingestFrame(now);
            ingestAccumNs += System.nanoTime() - t0;
            ingestCount++;
            presentReal();
            slotsPresented = 0;
            displayFrameCount++;
            sourceFrameCount++;
        } else {
            double progress = (lastFrameNanos != 0)
                    ? (now - lastFrameNanos) / frameIntervalNanos : 0.0;
            if (!glReady || !hasPrevFrame || lastFrameNanos == 0) {
                if (glReady) {
                    presentReal();
                    displayFrameCount++;
                }
            } else if (progress > stallThresh) {
                // Stream stalled: hold the last real frame, no synthesis.
                stallHoldCount++;
                presentReal();
                displayFrameCount++;
            } else {
                int maxSlots = multiplier - 1;
                // phaseAdvance lets a synthetic frame claim its vsync slot earlier,
                // so a jittery real-frame cadence is less likely to squeeze the
                // in-between frame out. 0.0 keeps the original timing.
                double effProgress = progress + phaseAdvance;
                int targetSlots = Math.min(maxSlots, (int) Math.floor(effProgress * multiplier));
                if (targetSlots > slotsPresented) {
                    int slot = slotsPresented + 1;
                    float phase = (float) slot / multiplier;
                    presentSynthetic(phase, now);
                    slotsPresented = slot;
                    displayFrameCount++;
                } else {
                    // No new slot this vsync: re-present the last frame so the
                    // back buffer always holds valid content (never garbage).
                    // A repeat, not a new displayed frame -> do NOT count it.
                    repeatPresentCount++;
                    if (slotsPresented == 0) {
                        // Slot gate not reached yet: the phase never crossed
                        // the threshold before the next real frame / this vsync.
                        gateMissCount++;
                    }
                    drawTextureToScreen(lastScreenTex);
                }
            }
        }

        // Always settle the FPS window, including the branches that used to
        // return early (stall / not-ready), which froze the overlay (A4).
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

        // Copy decoded frame into the curr texture. In pass-through mode this
        // is all ingest does: no luma pyramid, no motion estimation.
        copyExternalToTexture(videoTextureId, frameTexB, stMatrix);
        if (interpEnabled) {
            renderLuma(frameTexB, lumaTexB);
            if (engine == ENGINE_QCOM_ME && halfScaleMe && lumaHalfProgram != 0) {
                renderLumaHalf(frameTexB, lumaHalfB);
            }

            if (hasPrevFrame) {
                runMotionEstimation();
                if (meFailed) {
                    meFailCount++;
                }
            }
        }

        // Swap prev/curr
        int t = frameTexA; frameTexA = frameTexB; frameTexB = t;
        t = lumaTexA; lumaTexA = lumaTexB; lumaTexB = t;
        if (engine == ENGINE_QCOM_ME && halfScaleMe) {
            t = lumaHalfA; lumaHalfA = lumaHalfB; lumaHalfB = t;
        }
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
        meFailed = false;   // reset per real frame (A3: was write-only before)
        if (engine == ENGINE_QCOM_ME) {
            boolean okFull = false;
            try {
                okFull = nativeEstimateMotion(lumaTexA, lumaTexB, mvTexHw);
            } catch (Throwable t) {
                LimeLog.warning("QCOM motion estimation crashed: " + t.getMessage());
            }

            // Second estimator on half-resolution luma: identical hardware
            // search window covers twice the scene motion after scaling the
            // field back up. This is what keeps pans and fast movement inside
            // the search range on the Adreno block estimator.
            boolean okHalf = false;
            if (halfScaleMe && lumaHalfProgram != 0) {
                try {
                    okHalf = nativeEstimateMotion(lumaHalfA, lumaHalfB, mvTexHwHalf);
                } catch (Throwable t) {
                    LimeLog.warning("QCOM half-res motion estimation crashed: " + t.getMessage());
                }
                if (okHalf) {
                    halfScaleFailCount = 0;
                } else if (++halfScaleFailCount >= 8) {
                    // A single glitch used to disable range extension for the
                    // whole session (A8). Only a sustained failure does now.
                    halfScaleMe = false;
                    LimeLog.warning("QCOM half-res ME failed repeatedly, search-range extension disabled");
                }
            }

            if (okFull || okHalf) {
                if (fuseProgram != 0 && smoothProgram != 0 && mvRawFusedTex != 0) {
                    renderFusePass(okFull, okHalf);
                    renderSmoothPass();
                }

                // Backward field (curr -> prev) for forward-backward motion
                // consistency in the synthesis shader. A pixel is only
                // interpolated when the forward MV at its position and the
                // backward MV at the warped position agree (fwd + bwd ~ 0);
                // disagreement marks occlusion, where blending would ghost.
                // This is the third full-resolution HW ME pass per frame, so it
                // is gated by the user setting; OFF keeps the single-direction
                // gates (confidence / residual / divergence) only.
                if (fbcEnabled) {
                    boolean okBwd = false;
                    try {
                        okBwd = nativeEstimateMotion(lumaTexB, lumaTexA, mvTexBwd);
                    } catch (Throwable t) {
                        LimeLog.warning("QCOM backward motion estimation crashed: " + t.getMessage());
                    }
                    if (okBwd && mvBwdSm != 0) {
                        renderSmoothField(mvTexBwd, mvBwdSm);
                        bwdFieldValid = true;
                    } else {
                        bwdFieldValid = false;
                    }
                } else {
                    bwdFieldValid = false;
                }
            } else {
                meFailed = true;
                bwdFieldValid = false;
            }
        } else if (engine == ENGINE_GLSL) {
            // Coarse pass: prevMV (mvTexA) untouched this frame -> render into mvTexA
            renderMePass(meProgram, mvFboA, null, frameWidth, frameHeight);
            // Refine pass: read mvTexA, write mvTexB
            renderMePass(refineProgram, mvFboB, mvTexA, frameWidth, frameHeight);
        }
    }

    private void presentReal() {
        if (!glReady) {
            return;
        }
        // After the swap in ingestFrame(), frameTexA holds the newest frame.
        drawTextureToScreen(frameTexA != 0 ? frameTexA : frameTexB);
        realPresentCount++;
    }

    private void presentSynthetic(float phase, long now) {
        if (!glReady || !hasPrevFrame) {
            return;
        }
        if (meFailed) {
            // All HW motion passes failed for this frame pair: the MV field
            // still holds the previous pair's vectors. Hold the real frame
            // instead of synthesizing from stale motion (A3).
            drawTextureToScreen(frameTexA);
            realPresentCount++;
            return;
        }
        synthCount++;

        int mvTex;
        float mvMode;
        if (engine == ENGINE_QCOM_ME) {
            // Prefer the fused + smoothed multi-scale field; fall back to the
            // raw hardware field if fusion is unavailable.
            mvTex = (fuseProgram != 0 && smCurr != 0) ? smCurr : mvTexHw;
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
        // Magnitude sanity limit: the multi-scale (QCOM) field legitimately
        // carries motion well beyond the GLSL search range, so scale the cap
        // with the frame instead of clamping to the encoded-field range.
        float maxMv = (engine == ENGINE_QCOM_ME)
                ? Math.max(24.0f, frameWidth * 0.10f)
                : MV_RANGE * 4.0f;
        GLES20.glUniform1f(GLES20.glGetUniformLocation(synthProgram, "uMaxMV"), maxMv);
        GLES20.glUniform1f(GLES20.glGetUniformLocation(synthProgram, "uCutThresh"), cutThreshold);
        GLES20.glUniform2f(GLES20.glGetUniformLocation(synthProgram, "uMvGrid"),
                Math.max(1, mvGridW), Math.max(1, mvGridH));
        // Motion-field divergence gate. QCOM vectors are large (calibrated to
        // a fraction of the frame width); the GLSL fallback only searches
        // +/-MV_RANGE px, so it needs a proportionally small threshold or the
        // gate never fires at all (B2).
        float divThresh = (engine == ENGINE_QCOM_ME)
                ? Math.max(24.0f, frameWidth * 0.055f)
                : Math.max(4.0f, MV_RANGE * 1.5f);
        GLES20.glUniform1f(GLES20.glGetUniformLocation(synthProgram, "uDivThresh"),
                divThresh);

        // Backward motion field for forward-backward consistency (QCOM only;
        // the GLSL engine has no backward field and keeps its residual gates).
        boolean fbc = fbcEnabled && (engine == ENGINE_QCOM_ME) && bwdFieldValid && mvBwdSm != 0;
        GLES20.glActiveTexture(GLES20.GL_TEXTURE3);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, fbc ? mvBwdSm : frameTexA);
        GLES20.glUniform1i(GLES20.glGetUniformLocation(synthProgram, "uMvBwd"), 3);
        GLES20.glUniform1f(GLES20.glGetUniformLocation(synthProgram, "uFbcValid"), fbc ? 1.0f : 0.0f);
        GLES20.glUniform1f(GLES20.glGetUniformLocation(synthProgram, "uFbcThresh"), fbcThresh);

        // Render the synthetic frame into a texture, then present it through
        // the shared screen path (keeps viewport/letterbox handling uniform
        // and gives every vsync a valid back buffer).
        int outTex = getSynthTex();
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, synthFbo);
        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0, GLES20.GL_TEXTURE_2D, outTex, 0);
        GLES20.glViewport(0, 0, frameWidth, frameHeight);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
        for (int i = 0; i < 4; i++) {
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0 + i);
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);
        }
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);

        // Quality telemetry (H): accumulate sparse invalid-pixel samples.
        // Must run BEFORE the inpaint pass, which overwrites alpha with 1.
        renderMetrics(outTex);

        // Occlusion inpaint: the synthesis pass writes its per-pixel validity
        // into alpha; this pass fills fully-invalid pixels (disocclusions,
        // frame borders) with a validity-weighted average of the surrounding
        // valid pixels instead of leaving hard nearest-frame seams.
        if (inpaintProgram != 0) {
            int inTex = outTex;
            outTex = getSynthTex2();
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, synthFbo);
            GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0, GLES20.GL_TEXTURE_2D, outTex, 0);
            GLES20.glViewport(0, 0, frameWidth, frameHeight);
            GLES20.glUseProgram(inpaintProgram);
            bindQuad(inpaintProgram);
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, inTex);
            GLES20.glUniform1i(GLES20.glGetUniformLocation(inpaintProgram, "uSrc"), 0);
            GLES20.glUniform2f(GLES20.glGetUniformLocation(inpaintProgram, "uTexel"),
                    1.0f / frameWidth, 1.0f / frameHeight);
            GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
        }

        drawTextureToScreen(outTex);
        synthPresentCount++;
    }

    // (H) Additive-accumulation telemetry pass. Runs at METRICS_RES x
    // METRICS_RES (256 fragments), blending one sparse sample per frame
    // into a tiny RGBA16F target; updateFps() reads the sums back once per
    // second and clears the target for the next window. Blend state is
    // strictly scoped: every other pass in this renderer runs with
    // blending disabled and never sees it enabled.
    private void renderMetrics(int srcTex) {
        if (metricsProgram == 0) {
            return;
        }
        if (metricsTex == 0) {
            metricsTex = createRgba16fTexture(METRICS_RES, METRICS_RES);
            int[] fbos = new int[1];
            GLES20.glGenFramebuffers(1, fbos, 0);
            metricsFbo = fbos[0];
        }
        if (metricsFbo == 0 || metricsTex == 0) {
            return;
        }
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, metricsFbo);
        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0,
                GLES20.GL_TEXTURE_2D, metricsTex, 0);
        GLES20.glViewport(0, 0, METRICS_RES, METRICS_RES);
        GLES20.glUseProgram(metricsProgram);
        bindQuad(metricsProgram);
        GLES20.glEnable(GLES20.GL_BLEND);
        GLES20.glBlendFunc(GLES20.GL_ONE, GLES20.GL_ONE);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, srcTex);
        GLES20.glUniform1i(GLES20.glGetUniformLocation(metricsProgram, "uSrc"), 0);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
        GLES20.glDisable(GLES20.GL_BLEND);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
    }

    private int synthTex = 0;
    private int synthTex2 = 0;
    private int synthFbo = 0;
    private int lastScreenTex = 0;

    // Blur/softening (temporal persistence) resources. The presented frame is
    // blended with the previous presented frame: out = cur*(1-b) + prev*b.
    // Repeats of the same texture are presented raw so trails don't accumulate.
    private int blendProgram = 0;
    private final int[] blendTex = new int[2];   // double-buffered blend targets
    private int blendCurIdx = 0;                 // texture written this pass
    private int blendFbo = 0;
    private boolean blendHistoryValid = false;
    private final float[] identityMatrix = {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f};

    // Quality telemetry (H): sparse additive accumulation target + counters.
    private static final int METRICS_RES = 16;
    private int metricsProgram = 0;
    private int metricsTex = 0;
    private int metricsFbo = 0;
    private int synthCount = 0;
    private int meFailCount = 0;

    private int getSynthTex() {
        if (synthTex == 0) {
            synthTex = createColorTexture(frameWidth, frameHeight);
        }
        return synthTex;
    }

    private int getSynthTex2() {
        if (synthTex2 == 0) {
            synthTex2 = createColorTexture(frameWidth, frameHeight);
        }
        return synthTex2;
    }

    private void drawTextureToScreen(int tex) {
        if (tex == 0) {
            return;
        }
        applyScreenViewport();
        if (blurStrength > 0.0f && glReady
                && tex != lastScreenTex && tex != lastContentTex) {
            // New content: blend with the previous presented frame to soften
            // discrete frame steps (pan judder). Skipped for repeats — both
            // the idle re-present of lastScreenTex and re-presentation of the
            // same source texture (stream stall) — so trails never accumulate.
            int blended = renderBlendPass(tex);
            if (blended != 0) {
                tex = blended;
            }
        }
        lastContentTex = tex;
        lastScreenTex = tex;
        GLES20.glUseProgram(presentProgram);
        bindQuad(presentProgram);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, tex);
        GLES20.glUniform1i(GLES20.glGetUniformLocation(presentProgram, "s_InputTexture"), 0);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);
    }

    // Blend the current frame with the previous presented frame (temporal
    // persistence). Returns the blended texture to present, or 0 to present
    // the raw input (first frame, or shader/texture allocation failure).
    private int renderBlendPass(int curTex) {
        if (blendProgram == 0) {
            blendProgram = createProgram(VERT_SHADER, BLEND_FRAG);
            if (blendProgram == 0) {
                return 0;
            }
        }
        if (blendTex[0] == 0 && frameWidth > 0) {
            blendTex[0] = createColorTexture(frameWidth, frameHeight);
            blendTex[1] = createColorTexture(frameWidth, frameHeight);
            int[] fbo = new int[1];
            GLES20.glGenFramebuffers(1, fbo, 0);
            blendFbo = fbo[0];
        }
        if (blendTex[0] == 0 || blendTex[1] == 0 || blendFbo == 0) {
            return 0;
        }
        if (!blendHistoryValid) {
            // Seed the history with the current frame so the first real blend
            // is a visual no-op; this frame is presented raw.
            copyColorTex(curTex, blendTex[1 - blendCurIdx]);
            blendHistoryValid = true;
            return 0;
        }
        int dst = blendTex[blendCurIdx];
        int prev = blendTex[1 - blendCurIdx];
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, blendFbo);
        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0,
                GLES20.GL_TEXTURE_2D, dst, 0);
        GLES20.glViewport(0, 0, frameWidth, frameHeight);
        GLES20.glUseProgram(blendProgram);
        bindQuad(blendProgram);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, curTex);
        GLES20.glUniform1i(GLES20.glGetUniformLocation(blendProgram, "s_Cur"), 0);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE1);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, prev);
        GLES20.glUniform1i(GLES20.glGetUniformLocation(blendProgram, "s_Prev"), 1);
        GLES20.glUniform1f(GLES20.glGetUniformLocation(blendProgram, "uBlur"), blurStrength);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE1);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
        applyScreenViewport();
        // The texture just written becomes the history for the next frame.
        blendCurIdx = 1 - blendCurIdx;
        return dst;
    }

    // Plain color-texture -> color-texture copy (reuses copyProgram).
    private void copyColorTex(int srcTex, int dstTex) {
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, frameFbo);
        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0,
                GLES20.GL_TEXTURE_2D, dstTex, 0);
        GLES20.glViewport(0, 0, frameWidth, frameHeight);
        GLES20.glUseProgram(copyProgram);
        bindQuad(copyProgram);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, srcTex);
        GLES20.glUniform1i(GLES20.glGetUniformLocation(copyProgram, "s_InputTexture"), 0);
        GLES20.glUniformMatrix4fv(GLES20.glGetUniformLocation(copyProgram, "uTransform"),
                1, false, identityMatrix, 0);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
        applyScreenViewport();
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
        GLES20.glUniform2f(GLES20.glGetUniformLocation(lumaProgram, "uTexel"),
                1.0f / Math.max(1, frameWidth), 1.0f / Math.max(1, frameHeight));
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

    /**
     * Half-resolution luma pass used by the second hardware motion estimator.
     * 4-tap box average to keep the downsampled luma from aliasing, which
     * would otherwise feed false matches into the estimator.
     */
    private void renderLumaHalf(int srcTex, int destTex) {
        int hw = Math.max(1, frameWidth / 2);
        int hh = Math.max(1, frameHeight / 2);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, lumaHalfFbo);
        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0, GLES20.GL_TEXTURE_2D, destTex, 0);
        GLES20.glViewport(0, 0, hw, hh);
        GLES20.glUseProgram(lumaHalfProgram);
        bindQuad(lumaHalfProgram);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, srcTex);
        GLES20.glUniform1i(GLES20.glGetUniformLocation(lumaHalfProgram, "s_InputTexture"), 0);
        GLES20.glUniform2f(GLES20.glGetUniformLocation(lumaHalfProgram, "uSrcTexel"),
                1.0f / frameWidth, 1.0f / frameHeight);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
    }

    /**
     * Picks, per block, the motion candidate that matches the best among:
     *   1. full-res hardware field,
     *   2. half-res hardware field scaled x2 (doubles the search range),
     *   3. previous frame's field (temporal prediction for smooth pans).
     * Candidates are scored with a 5-tap SAD on the luma pair, so a candidate
     * that lands on the wrong content loses regardless of its origin.
     */
    private void renderFusePass(boolean fullValid, boolean halfValid) {
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fuseFbo);
        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0, GLES20.GL_TEXTURE_2D, mvRawFusedTex, 0);
        GLES20.glViewport(0, 0, mvGridW, mvGridH);
        GLES20.glUseProgram(fuseProgram);
        bindQuad(fuseProgram);

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, lumaTexB);
        GLES20.glUniform1i(GLES20.glGetUniformLocation(fuseProgram, "uCurrLuma"), 0);

        GLES20.glActiveTexture(GLES20.GL_TEXTURE1);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, lumaTexA);
        GLES20.glUniform1i(GLES20.glGetUniformLocation(fuseProgram, "uPrevLuma"), 1);

        GLES20.glActiveTexture(GLES20.GL_TEXTURE2);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, mvTexHw);
        GLES20.glUniform1i(GLES20.glGetUniformLocation(fuseProgram, "uMvFull"), 2);

        GLES20.glActiveTexture(GLES20.GL_TEXTURE3);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, mvTexHwHalf);
        GLES20.glUniform1i(GLES20.glGetUniformLocation(fuseProgram, "uMvHalf"), 3);

        GLES20.glActiveTexture(GLES20.GL_TEXTURE4);
        // smCurr still holds the previous frame's field at this point (the
        // swap happens in renderSmoothPass), i.e. exactly the temporal
        // predictor we want for smooth pans.
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, smCurr);
        GLES20.glUniform1i(GLES20.glGetUniformLocation(fuseProgram, "uMvPrev"), 4);

        GLES20.glUniform2f(GLES20.glGetUniformLocation(fuseProgram, "uFrameSize"), frameWidth, frameHeight);
        GLES20.glUniform2f(GLES20.glGetUniformLocation(fuseProgram, "uGridFull"), mvGridW, mvGridH);
        GLES20.glUniform2f(GLES20.glGetUniformLocation(fuseProgram, "uBlock"), mvBlockX, mvBlockY);
        GLES20.glUniform1f(GLES20.glGetUniformLocation(fuseProgram, "uHalfValid"), halfValid ? 1.0f : 0.0f);
        GLES20.glUniform1f(GLES20.glGetUniformLocation(fuseProgram, "uFullValid"), fullValid ? 1.0f : 0.0f);
        GLES20.glUniform1f(GLES20.glGetUniformLocation(fuseProgram, "uPrevValid"), temporalPred ? 1.0f : 0.0f);

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
        for (int i = 0; i < 5; i++) {
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0 + i);
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);
        }
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
    }

    /**
     * 3x3 component-wise median over the fused field: removes isolated wrong
     * blocks that would otherwise smear a single object across the frame.
     * The result becomes the current field; the old current field stays
     * available as next frame's temporal predictor.
     */
    private void renderSmoothPass() {
        // The other buffer holds a stale field that nothing reads, so it is
        // reused for the new one.
        int dst = smPrev;
        renderSmoothField(mvRawFusedTex, dst);

        // New field becomes current, old current becomes the predictor.
        int t = smPrev;
        smPrev = smCurr;
        smCurr = t;
        temporalPred = true;
    }

    /**
     * 3x3 median smoothing of any block-grid MV field. Shared by the forward
     * fused field and the backward (consistency) field.
     */
    private void renderSmoothField(int srcTex, int dstTex) {
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, smoothFbo);
        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0, GLES20.GL_TEXTURE_2D, dstTex, 0);
        GLES20.glViewport(0, 0, mvGridW, mvGridH);
        GLES20.glUseProgram(smoothProgram);
        bindQuad(smoothProgram);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, srcTex);
        GLES20.glUniform1i(GLES20.glGetUniformLocation(smoothProgram, "uMvRaw"), 0);
        GLES20.glUniform2f(GLES20.glGetUniformLocation(smoothProgram, "uGrid"), mvGridW, mvGridH);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
    }

    private void allocateFrameResources() {
        frameTexA = createColorTexture(frameWidth, frameHeight);
        frameTexB = createColorTexture(frameWidth, frameHeight);
        if (interpEnabled) {
            lumaTexA = createR8Texture(frameWidth, frameHeight);
            lumaTexB = createR8Texture(frameWidth, frameHeight);
        }

        int[] fbos = new int[1];
        GLES20.glGenFramebuffers(1, fbos, 0);
        frameFbo = fbos[0];
        if (interpEnabled) {
            GLES20.glGenFramebuffers(1, fbos, 0);
            lumaFbo = fbos[0];
        }
        GLES20.glGenFramebuffers(1, fbos, 0);
        synthFbo = fbos[0];

        if (interpEnabled && engine == ENGINE_GLSL) {
            int mvW = (frameWidth + MV_GRID - 1) / MV_GRID;
            int mvH = (frameHeight + MV_GRID - 1) / MV_GRID;
            mvGridW = mvW;
            mvGridH = mvH;
            mvTexA = createColorTexture(mvW, mvH);
            mvTexB = createColorTexture(mvW, mvH);
            GLES20.glGenFramebuffers(1, fbos, 0);
            mvFboA = fbos[0];
            GLES20.glGenFramebuffers(1, fbos, 0);
            mvFboB = fbos[0];
        } else if (interpEnabled && engine == ENGINE_QCOM_ME) {
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
            mvGridW = (frameWidth + mvBlockX - 1) / mvBlockX;
            mvGridH = (frameHeight + mvBlockY - 1) / mvBlockY;

            int halfW = Math.max(1, frameWidth / 2);
            int halfH = Math.max(1, frameHeight / 2);
            mvTexHw = createRgba16fTexture(mvGridW, mvGridH);
            mvTexHwHalf = createRgba16fTexture(
                    (halfW + mvBlockX - 1) / mvBlockX,
                    (halfH + mvBlockY - 1) / mvBlockY);
            mvTexBwd = createRgba16fTexture(mvGridW, mvGridH);

            lumaHalfA = createR8Texture(halfW, halfH);
            lumaHalfB = createR8Texture(halfW, halfH);
            GLES20.glGenFramebuffers(1, fbos, 0);
            lumaHalfFbo = fbos[0];
            GLES20.glGenFramebuffers(1, fbos, 0);
            fuseFbo = fbos[0];
            GLES20.glGenFramebuffers(1, fbos, 0);
            smoothFbo = fbos[0];

            // Zero-init every field: mv = 0, conf = 0 -> synthesis holds the
            // nearest real frame until a valid field exists (uninitialised
            // RGBA16F memory would otherwise inject garbage motion).
            clearRgba16fTexture(mvTexHw, mvGridW, mvGridH);
            clearRgba16fTexture(mvTexHwHalf,
                    (halfW + mvBlockX - 1) / mvBlockX,
                    (halfH + mvBlockY - 1) / mvBlockY);

            if (isFboRenderable(mvTexBwd, mvGridW, mvGridH)) {
                mvBwdSm = createRgba16fTexture(mvGridW, mvGridH);
                clearRgba16fTexture(mvBwdSm, mvGridW, mvGridH);
                // Consistency tolerance scales with the block size: fields are
                // block-quantised, so even perfect motion disagrees by up to
                // ~half a block between the two directions.
                fbcThresh = Math.max(6.0f, Math.max(mvBlockX, mvBlockY) * 0.45f);
            } else {
                mvBwdSm = 0;
                bwdFieldValid = false;
            }

            if (fuseProgram != 0 && smoothProgram != 0) {
                mvRawFusedTex = createRgba16fTexture(mvGridW, mvGridH);
                mvSmA = createRgba16fTexture(mvGridW, mvGridH);
                mvSmB = createRgba16fTexture(mvGridW, mvGridH);
                smCurr = mvSmA;
                smPrev = mvSmB;

                // RGBA16F render targets are not guaranteed on every driver
                // (needs EXT_color_buffer_float / _half_float). Verify one
                // attachment; if incomplete, drop fusion and use the raw
                // hardware field with the legacy presentation path.
                if (!isFboRenderable(mvRawFusedTex, mvGridW, mvGridH)) {
                    LimeLog.warning("RGBA16F motion field not renderable, multi-scale fusion disabled");
                    mvRawFusedTex = 0;
                    mvSmA = 0;
                    mvSmB = 0;
                    smCurr = 0;
                    smPrev = 0;
                    halfScaleMe = false;
                    temporalPred = false;
                } else {
                    clearRgba16fTexture(mvSmA, mvGridW, mvGridH);
                    clearRgba16fTexture(mvSmB, mvGridW, mvGridH);
                    temporalPred = false;
                }
            }
            LimeLog.info("Frame interpolation: QCOM ME block " + mvBlockX + "x" + mvBlockY
                    + ", grid " + mvGridW + "x" + mvGridH + ", multiscale="
                    + (mvRawFusedTex != 0) + ", fbc=" + fbcEnabled);
        }
    }

    private boolean isFboRenderable(int tex, int w, int h) {
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fuseFbo);
        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0, GLES20.GL_TEXTURE_2D, tex, 0);
        GLES20.glViewport(0, 0, w, h);
        int status = GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
        return status == GLES20.GL_FRAMEBUFFER_COMPLETE;
    }

    private void clearRgba16fTexture(int tex, int w, int h) {
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, smoothFbo);
        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0, GLES20.GL_TEXTURE_2D, tex, 0);
        GLES20.glViewport(0, 0, w, h);
        GLES20.glClearColor(0f, 0f, 0f, 0f);
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
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
            int realFrames = Math.max(1, sourceFrameCount);
            float meFailPct = 100.0f * meFailCount / realFrames;
            float seconds = elapsed / 1_000_000_000f;
            float loopHz = loopFrameCount / seconds;
            float realFps = realPresentCount / seconds;
            float synthFps = synthPresentCount / seconds;
            float repeatFps = repeatPresentCount / seconds;
            float gateMissFps = gateMissCount / seconds;
            float stallFps = stallHoldCount / seconds;
            float ingestAvgMs = (ingestCount > 0)
                    ? (float) (ingestAccumNs / 1e6 / ingestCount) : -1.0f;
            displayFrameCount = 0;
            sourceFrameCount = 0;
            loopFrameCount = 0;
            realPresentCount = 0;
            synthPresentCount = 0;
            repeatPresentCount = 0;
            gateMissCount = 0;
            stallHoldCount = 0;
            fpsWindowStart = now;

            // (H) once-per-second telemetry readback: 256 texels from the
            // additive accumulation target, then reset it for the next
            // window. glReadPixels of RGBA16F as GL_FLOAT works on any
            // driver that renders RGBA16F (EXT_color_buffer_float); if the
            // format pair is refused the buffer stays zero and the metric
            // is skipped for that window — logging only, never fatal.
            float invalidRate = -1.0f;
            if (metricsFbo != 0 && metricsTex != 0) {
                java.nio.FloatBuffer buf = java.nio.ByteBuffer
                        .allocateDirect(METRICS_RES * METRICS_RES * 16)
                        .order(java.nio.ByteOrder.nativeOrder()).asFloatBuffer();
                GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, metricsFbo);
                GLES20.glReadPixels(0, 0, METRICS_RES, METRICS_RES,
                        GLES20.GL_RGBA, GLES20.GL_FLOAT, buf);
                GLES20.glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
                GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);
                GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
                buf.rewind();
                double invSum = 0.0, cntSum = 0.0;
                for (int i = 0; i < METRICS_RES * METRICS_RES; i++) {
                    invSum += buf.get();
                    cntSum += buf.get();
                }
                if (cntSum > 0.5) {
                    invalidRate = (float) (invSum / cntSum);
                }
            }

            String line = "Frame interpolation: display=" + String.format("%.1f", displayFps)
                    + " source=" + String.format("%.1f", sourceFps)
                    + String.format(" loop=%.1fHz", loopHz)
                    + String.format(" real=%.1f synth=%.1f", realFps, synthFps)
                    + String.format(" repeat=%.1f gateMiss=%.1f stall=%.1f",
                            repeatFps, gateMissFps, stallFps)
                    + " engine=" + engineString() + " x" + multiplier
                    + String.format(" phase=%.2f", phaseAdvance)
                    + String.format(" meFail=%.0f%%", meFailPct)
                    + String.format(" synth=%d", synthCount)
                    + (invalidRate >= 0.0f ? String.format(" invalid=%.1f%%", invalidRate * 100.0f) : "")
                    + (ingestAvgMs >= 0.0f ? String.format(" ingestAvg=%.1fms", ingestAvgMs) : "");
            LimeLog.info(line);
            writeDiagLine(now, displayFps, sourceFps, loopHz, realFps, synthFps,
                    repeatFps, gateMissFps, stallFps, meFailPct, invalidRate, ingestAvgMs);
            ingestAccumNs = 0;
            ingestCount = 0;
            meFailCount = 0;
            synthCount = 0;
        }
    }

    // Mirror the once-per-second diagnostics line to
    // /sdcard/Android/data/<pkg>/files/frame_interp_log.txt (one line per
    // second, session headers, 8 MB rotation). Failures are swallowed:
    // diagnostics must never break rendering.
    private void writeDiagLine(long now, float disp, float src, float loopHz,
                               float realFps, float synthFps, float repeatFps,
                               float gateMissFps, float stallFps, float meFailPct,
                               float invalidRate, float ingestAvgMs) {
        if (!diagLogEnabled) {
            return;
        }
        try {
            if (logContext == null) {
                return;
            }
            if (diagFile == null) {
                File dir = logContext.getExternalFilesDir(null);
                if (dir == null) {
                    return;
                }
                diagFile = new File(dir, "frame_interp_log.txt");
            }
            boolean rotate = diagFile.exists() && diagFile.length() > DIAG_LOG_MAX_BYTES;
            if (diagWriter == null || rotate) {
                if (diagWriter != null) {
                    try {
                        diagWriter.close();
                    } catch (Exception ignored) {
                    }
                    diagWriter = null;
                }
                boolean append = !rotate && !diagSessionHeader
                        && diagFile.exists() && diagFile.length() > 0;
                diagWriter = new BufferedWriter(new FileWriter(diagFile, append));
                diagSessionHeader = true;
                String header = "# session start "
                        + new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(new Date());
                diagWriter.write(header);
                diagWriter.newLine();
            }
            // Wall-clock timestamp: `now` is System.nanoTime() (arbitrary
            // origin), so it must NOT be fed to Date — it produced nonsense
            // times like 13:08:51 for a 16:28 session.
            String ts = new SimpleDateFormat("HH:mm:ss.SSS", Locale.US).format(new Date());
            String line = ts
                    + " disp=" + String.format("%.1f", disp)
                    + " src=" + String.format("%.1f", src)
                    + " loop=" + String.format("%.1f", loopHz) + "Hz"
                    + " real=" + String.format("%.1f", realFps)
                    + " synth=" + String.format("%.1f", synthFps)
                    + " rep=" + String.format("%.1f", repeatFps)
                    + " gateMiss=" + String.format("%.1f", gateMissFps)
                    + " stall=" + String.format("%.1f", stallFps)
                    + " meFail=" + String.format("%.0f%%", meFailPct)
                    + (invalidRate >= 0.0f ? " invalid=" + String.format("%.1f%%", invalidRate * 100.0f) : "")
                    + (ingestAvgMs >= 0.0f ? " ingest=" + String.format("%.1fms", ingestAvgMs) : "");
            diagWriter.write(line);
            diagWriter.newLine();
            diagWriter.flush();
        } catch (Exception e) {
            LimeLog.warning("Frame interp diag log: " + e.getMessage());
            if (diagWriter != null) {
                try {
                    diagWriter.close();
                } catch (Exception ignored) {
                }
                diagWriter = null;
            }
        }
    }

    private void closeDiagLog() {
        if (diagWriter != null) {
            try {
                diagWriter.close();
            } catch (Exception ignored) {
            }
            diagWriter = null;
        }
    }

    public static String engineString() {
        switch (engine) {
            case ENGINE_QCOM_ME: return "QCOM-ME";
            case ENGINE_GLSL: return "GLSL";
            default: return "OFF";
        }
    }

    public void onSurfaceDestroyed() {
        LimeLog.info("Shutting down frame interpolation renderer.");
        closeDiagLog();
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

        // Snapshot every GL id into final locals BEFORE queueing the async
        // cleanup (A1 + race fix). The old lambda referenced instance fields,
        // so a fast surface recreation could make it delete the NEW resources.
        final int[] progSnap = {copyProgram, lumaProgram, lumaHalfProgram, meProgram,
                refineProgram, fuseProgram, smoothProgram, synthProgram,
                presentProgram, inpaintProgram, metricsProgram, blendProgram};
        final int[] texSnap = {videoTextureId, frameTexA, frameTexB, lumaTexA, lumaTexB,
                lumaHalfA, lumaHalfB, mvTexA, mvTexB, mvTexHw, mvTexHwHalf,
                mvTexBwd, mvBwdSm, mvRawFusedTex, mvSmA, mvSmB, synthTex, synthTex2, metricsTex,
                blendTex[0], blendTex[1]};
        final int[] fboSnap = {frameFbo, lumaFbo, lumaHalfFbo, synthFbo,
                mvFboA, mvFboB, fuseFbo, smoothFbo, metricsFbo, blendFbo};

        if (glSurfaceView != null) {
            glSurfaceView.queueEvent(() -> {
                // De-duplicate program ids before deletion.
                java.util.HashSet<Integer> seen = new java.util.HashSet<>();
                for (int p : progSnap) {
                    if (p != 0 && seen.add(p)) {
                        GLES20.glDeleteProgram(p);
                    }
                }
                int count = 0;
                int[] compact = new int[texSnap.length];
                for (int t : texSnap) {
                    if (t != 0) compact[count++] = t;
                }
                if (count > 0) {
                    GLES20.glDeleteTextures(count, compact, 0);
                }
                int fCount = 0;
                int[] fCompact = new int[fboSnap.length];
                for (int f : fboSnap) {
                    if (f != 0) fCompact[fCount++] = f;
                }
                if (fCount > 0) {
                    GLES20.glDeleteFramebuffers(fCount, fCompact, 0);
                }
            });
        }

        // Full state reset (A5). Cadence/stat fields were previously left
        // stale, corrupting pacing and FPS right after a surface rebuild.
        glReady = false;
        isActive = false;
        hasPrevFrame = false;
        frameWidth = 0;
        frameHeight = 0;
        surfaceWidth = 0;
        surfaceHeight = 0;
        lastScreenTex = 0;
        lastFrameNanos = 0;
        frameIntervalNanos = defaultFrameIntervalNanos();
        slotsPresented = 0;
        fpsWindowStart = 0;
        displayFrameCount = 0;
        sourceFrameCount = 0;
        ingestAccumNs = 0;
        ingestCount = 0;
        meFailed = false;
        halfScaleMe = true;
        halfScaleFailCount = 0;
        temporalPred = false;
        bwdFieldValid = false;
        fbcThresh = 6.0f;
        engine = ENGINE_NONE;
        displayFps = 0;
        sourceFps = 0;
        mvGridW = 0;
        mvGridH = 0;

        // Clear every live id so a rebuilt surface re-allocates from scratch
        // (getSynthTex() is lazy and would otherwise reuse a deleted handle).
        copyProgram = 0; lumaProgram = 0; lumaHalfProgram = 0; meProgram = 0;
        refineProgram = 0; fuseProgram = 0; smoothProgram = 0; synthProgram = 0;
        presentProgram = 0; inpaintProgram = 0;
        videoTextureId = 0;
        frameTexA = 0; frameTexB = 0; lumaTexA = 0; lumaTexB = 0;
        lumaHalfA = 0; lumaHalfB = 0; mvTexA = 0; mvTexB = 0;
        mvTexHw = 0; mvTexHwHalf = 0; mvTexBwd = 0; mvBwdSm = 0;
        mvRawFusedTex = 0; mvSmA = 0; mvSmB = 0;
        smCurr = 0; smPrev = 0;
        synthTex = 0; synthTex2 = 0; synthFbo = 0;
        frameFbo = 0; lumaFbo = 0; lumaHalfFbo = 0;
        mvFboA = 0; mvFboB = 0; fuseFbo = 0; smoothFbo = 0;
        metricsProgram = 0; metricsTex = 0; metricsFbo = 0;
        blendProgram = 0; blendTex[0] = 0; blendTex[1] = 0;
        blendFbo = 0; blendCurIdx = 0; blendHistoryValid = false;
        lastContentTex = 0;
        synthCount = 0; meFailCount = 0;
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

    // Full-resolution luma fed to the hardware motion estimators. A very
    // light 3x3 Gaussian runs first: codec blocking noise pollutes the
    // block-matching SAD and makes the estimator flicker on dark and
    // gradient scenes. The blur touches only the estimation input —
    // synthesis always samples the untouched frame textures — so the
    // visible image quality is never affected by it.
    private static final String LUMA_FRAG =
            "#version 300 es\n" +
            "precision mediump float;\n" +
            "uniform mediump sampler2D s_InputTexture;\n" +
            "uniform vec2 uTexel;\n" +
            "in vec2 vUV;\n" +
            "out vec4 outColor;\n" +
            "void main() {\n" +
            "    vec3 acc = texture(s_InputTexture, vUV).rgb * 4.0;\n" +
            "    acc += (texture(s_InputTexture, vUV + vec2(-uTexel.x, 0.0)).rgb\n" +
            "          + texture(s_InputTexture, vUV + vec2(uTexel.x, 0.0)).rgb\n" +
            "          + texture(s_InputTexture, vUV + vec2(0.0, -uTexel.y)).rgb\n" +
            "          + texture(s_InputTexture, vUV + vec2(0.0, uTexel.y)).rgb) * 2.0;\n" +
            "    acc += texture(s_InputTexture, vUV + vec2(-uTexel.x, -uTexel.y)).rgb\n" +
            "         + texture(s_InputTexture, vUV + vec2(uTexel.x, -uTexel.y)).rgb\n" +
            "         + texture(s_InputTexture, vUV + vec2(-uTexel.x, uTexel.y)).rgb\n" +
            "         + texture(s_InputTexture, vUV + vec2(uTexel.x, uTexel.y)).rgb;\n" +
            "    vec3 c = acc / 16.0;\n" +
            "    float l = 0.299 * c.r + 0.587 * c.g + 0.114 * c.b;\n" +
            "    outColor = vec4(l, 0.0, 0.0, 1.0);\n" +
            "}\n";

    // Half-res luma for the second (range-extended) hardware estimator.
    // 4-tap box average keeps the downsample from aliasing, which would feed
    // false matches into the estimator.
    private static final String LUMA_HALF_FRAG =
            "#version 300 es\n" +
            "precision mediump float;\n" +
            "uniform mediump sampler2D s_InputTexture;\n" +
            "uniform vec2 uSrcTexel;\n" +
            "in vec2 vUV;\n" +
            "out vec4 outColor;\n" +
            "void main() {\n" +
            "    vec3 c = texture(s_InputTexture, vUV + vec2(-0.5, -0.5) * uSrcTexel).rgb;\n" +
            "    c += texture(s_InputTexture, vUV + vec2(0.5, -0.5) * uSrcTexel).rgb;\n" +
            "    c += texture(s_InputTexture, vUV + vec2(-0.5, 0.5) * uSrcTexel).rgb;\n" +
            "    c += texture(s_InputTexture, vUV + vec2(0.5, 0.5) * uSrcTexel).rgb;\n" +
            "    c *= 0.25;\n" +
            "    float l = 0.299 * c.r + 0.587 * c.g + 0.114 * c.b;\n" +
            "    outColor = vec4(l, 0.0, 0.0, 1.0);\n" +
            "}\n";

    // Multi-scale candidate fusion (QCOM path), one fragment per ME block.
    // Candidates: full-res field, half-res field x2, previous frame's field.
    // Each is scored with a 5-tap SAD on the luma pair at the block centre;
    // the winner's score becomes the confidence. A candidate that lands on
    // the wrong content loses no matter where it came from, so a bad
    // half-res match can never replace a good full-res one.
    private static final String FUSE_FRAG =
            "#version 300 es\n" +
            "precision highp float;\n" +
            "uniform mediump sampler2D uCurrLuma;\n" +
            "uniform mediump sampler2D uPrevLuma;\n" +
            "uniform highp sampler2D uMvFull;\n" +
            "uniform highp sampler2D uMvHalf;\n" +
            "uniform highp sampler2D uMvPrev;\n" +
            "uniform vec2 uFrameSize;\n" +
            "uniform vec2 uGridFull;\n" +
            "uniform vec2 uBlock;\n" +
            "uniform float uFullValid;\n" +
            "uniform float uHalfValid;\n" +
            "uniform float uPrevValid;\n" +
            "in vec2 vUV;\n" +
            "out vec4 outColor;\n" +
            "\n" +
            "float tapDiff(vec2 curPx, vec2 mv) {\n" +
            "    vec2 cc = clamp(curPx / uFrameSize, vec2(0.0), vec2(1.0));\n" +
            "    vec2 pc = clamp((curPx - mv) / uFrameSize, vec2(0.0), vec2(1.0));\n" +
            "    return abs(texture(uCurrLuma, cc).r - texture(uPrevLuma, pc).r);\n" +
            "}\n" +
            "\n" +
            "float sad5(vec2 c, vec2 mv) {\n" +
            "    vec2 q = uBlock * 0.35;\n" +
            "    return tapDiff(c, mv)\n" +
            "         + tapDiff(c + vec2(q.x, 0.0), mv)\n" +
            "         + tapDiff(c - vec2(q.x, 0.0), mv)\n" +
            "         + tapDiff(c + vec2(0.0, q.y), mv)\n" +
            "         + tapDiff(c - vec2(0.0, q.y), mv);\n" +
            "}\n" +
            "\n" +
            "void main() {\n" +
            "    vec2 blockUv = (floor(vUV * uGridFull) + 0.5) / uGridFull;\n" +
            "    vec2 centerPx = blockUv * uFrameSize;\n" +
            "\n" +
            "    vec2 mvFull = texture(uMvFull, vUV).rg;\n" +
            "    vec2 mvHalf = texture(uMvHalf, vUV).rg * 2.0;\n" +
            "    vec2 mvPrev = texture(uMvPrev, vUV).rg;\n" +
            "\n" +
            "    float sFull = (uFullValid > 0.5) ? sad5(centerPx, mvFull) : 1e6;\n" +
            "    float sHalf = (uHalfValid > 0.5) ? sad5(centerPx, mvHalf) : 1e6;\n" +
            "    float sPrev = (uPrevValid > 0.5) ? sad5(centerPx, mvPrev) : 1e6;\n" +
            "\n" +
            "    float best = sFull;\n" +
            "    vec2 bestMv = mvFull;\n" +
            "    if (sHalf < best) { best = sHalf; bestMv = mvHalf; }\n" +
            "    if (sPrev < best) { best = sPrev; bestMv = mvPrev; }\n" +
            "\n" +
            "    // 5 taps, ~0.15 luma tolerance each.\n" +
            "    float conf = clamp(1.0 - best / 0.75, 0.0, 1.0);\n" +
            "    outColor = vec4(bestMv, conf, 1.0);\n" +
            "}\n";

    // 3x3 component-wise median over the fused field: kills isolated wrong
    // blocks that would otherwise smear one object across the frame.
    private static final String SMOOTH_FRAG =
            "#version 300 es\n" +
            "precision highp float;\n" +
            "uniform highp sampler2D uMvRaw;\n" +
            "uniform vec2 uGrid;\n" +
            "in vec2 vUV;\n" +
            "out vec4 outColor;\n" +
            "\n" +
            "vec2 samp(vec2 uv) {\n" +
            "    return texture(uMvRaw, clamp(uv, 0.5 / uGrid, 1.0 - 0.5 / uGrid)).rg;\n" +
            "}\n" +
            "float med3(float a, float b, float c) {\n" +
            "    return max(min(a, b), min(max(a, b), c));\n" +
            "}\n" +
            "vec2 med3v(vec2 a, vec2 b, vec2 c) {\n" +
            "    return vec2(med3(a.x, b.x, c.x), med3(a.y, b.y, c.y));\n" +
            "}\n" +
            "\n" +
            "void main() {\n" +
            "    vec2 t = 1.0 / uGrid;\n" +
            "    vec2 r0 = med3v(samp(vUV + vec2(-t.x, -t.y)), samp(vUV + vec2(0.0, -t.y)), samp(vUV + vec2(t.x, -t.y)));\n" +
            "    vec2 r1 = med3v(samp(vUV + vec2(-t.x, 0.0)), samp(vUV), samp(vUV + vec2(t.x, 0.0)));\n" +
            "    vec2 r2 = med3v(samp(vUV + vec2(-t.x, t.y)), samp(vUV + vec2(0.0, t.y)), samp(vUV + vec2(t.x, t.y)));\n" +
            "    vec2 m = med3v(r0, r1, r2);\n" +
            "    outColor = vec4(m, texture(uMvRaw, vUV).b, 1.0);\n" +
            "}\n";

    // Temporal persistence blend: out = cur*(1-b) + prev*b. Smears discrete
    // frame steps into short motion trails, reducing perceived judder for
    // pan shots whose cadence does not divide the refresh rate.
    private static final String BLEND_FRAG =
            "#version 300 es\n" +
            "precision mediump float;\n" +
            "uniform mediump sampler2D s_Cur;\n" +
            "uniform mediump sampler2D s_Prev;\n" +
            "uniform mediump float uBlur;\n" +
            "in vec2 vUV;\n" +
            "out vec4 outColor;\n" +
            "void main() {\n" +
            "    vec3 c = texture(s_Cur, vUV).rgb;\n" +
            "    vec3 p = texture(s_Prev, vUV).rgb;\n" +
            "    outColor = vec4(mix(c, p, uBlur), 1.0);\n" +
            "}\n";

    private static final String PRESENT_FRAG =
            "#version 300 es\n" +
            "precision mediump float;\n" +
            "uniform mediump sampler2D s_InputTexture;\n" +
            "in vec2 vUV;\n" +
            "out vec4 outColor;\n" +
            "void main() {\n" +
            "    vec4 c = texture(s_InputTexture, vUV);\n" +
            "    // Synthesis stores per-pixel validity in alpha; force opaque\n" +
            "    // so a missing inpaint pass can never show transparent holes.\n" +
            "    outColor = vec4(c.rgb, 1.0);\n" +
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

    // Bidirectional synthesis with validity gating.
    // uPhase in (0,1): 0 = prev, 1 = curr.
    // uMvMode 0: encoded RGBA8 field (GLSL path); 1: raw pixels in RGBA16F (QCOM).
    //
    // The warped prev/curr samples are only blended where the motion is
    // trustworthy, verified by four independent gates:
    //   1. estimator confidence,
    //   2. post-compensation residual between the warped samples,
    //   3. MV-field divergence (object boundaries),
    //   4. forward-backward consistency: the forward MV here and the backward
    //      MV at the warped position must agree (fwd + bwd ~ 0); disagreement
    //      is the classic disocclusion signature (FSR3-style check).
    // Pixels identical in both real frames (HUD, crosshair, letterboxed
    // background) bypass motion entirely and copy the current frame, so the
    // estimator can never warp static UI. Per-pixel validity is written to
    // alpha for the occlusion inpaint pass.
    private static final String SYNTH_FRAG =
            "#version 300 es\n" +
            "precision highp float;\n" +
            "uniform mediump sampler2D uCurr;\n" +
            "uniform mediump sampler2D uPrev;\n" +
            "uniform highp sampler2D uMV;\n" +
            "uniform highp sampler2D uMvBwd;\n" +
            "uniform float uPhase;\n" +
            "uniform float uMvMode;\n" +
            "uniform float uMvRange;\n" +
            "uniform vec2 uFrameSize;\n" +
            "uniform float uMaxMV;\n" +
            "uniform float uCutThresh;\n" +
            "uniform vec2 uMvGrid;\n" +
            "uniform float uDivThresh;\n" +
            "uniform float uFbcValid;\n" +
            "uniform float uFbcThresh;\n" +
            "in vec2 vUV;\n" +
            "out vec4 outColor;\n" +
            "\n" +
            "vec2 decodeMv(vec2 raw) {\n" +
            "    return mix((raw - 0.5) * uMvRange, raw, uMvMode);\n" +
            "}\n" +
            "\n" +
            "void main() {\n" +
            "    vec4 m = texture(uMV, vUV);\n" +
            "    vec2 mvPx = decodeMv(m.rg);\n" +
            "    float conf = m.b;\n" +
            "\n" +
            "    // GLSL ES 3.00 has no isnan(). A NaN fails every comparison\n" +
            "    // (x==x is false, x<k is false), so invert the magnitude test\n" +
            "    // to reject NaN/Inf and out-of-range vectors in one branch.\n" +
            "    conf = (conf == conf) ? clamp(conf, 0.0, 1.0) : 0.0;\n" +
            "    if (!(length(mvPx) < uMaxMV)) {\n" +
            "        mvPx = vec2(0.0);\n" +
            "        conf = 0.0;\n" +
            "    }\n" +
            "\n" +
            "    // MV-field divergence: a jump against the neighbours marks an\n" +
            "    // object boundary / occlusion, where any blend would ghost.\n" +
            "    vec2 t = 1.0 / uMvGrid;\n" +
            "    vec2 mR = decodeMv(texture(uMV, vUV + vec2(t.x, 0.0)).rg);\n" +
            "    vec2 mL = decodeMv(texture(uMV, vUV - vec2(t.x, 0.0)).rg);\n" +
            "    vec2 mU = decodeMv(texture(uMV, vUV + vec2(0.0, t.y)).rg);\n" +
            "    vec2 mD = decodeMv(texture(uMV, vUV - vec2(0.0, t.y)).rg);\n" +
            "    float div = length(mR - mL) + length(mU - mD);\n" +
            "\n" +
            "    vec2 mvUv = mvPx / uFrameSize;\n" +
            "    vec2 prevUv = vUV - mvUv * uPhase;\n" +
            "    vec2 currUv = vUV + mvUv * (1.0 - uPhase);\n" +
            "\n" +
            "    vec2 pc = clamp(prevUv, vec2(0.0), vec2(1.0));\n" +
            "    vec2 cc = clamp(currUv, vec2(0.0), vec2(1.0));\n" +
            "    float outside = (any(notEqual(pc, prevUv)) || any(notEqual(cc, currUv))) ? 1.0 : 0.0;\n" +
            "\n" +
            "    vec4 cPrev = texture(uPrev, pc);\n" +
            "    vec4 cCurr = texture(uCurr, cc);\n" +
            "\n" +
            "    // Residual after motion compensation: do both frames agree here?\n" +
            "    float res = dot(abs(cPrev.rgb - cCurr.rgb), vec3(0.3333));\n" +
            "\n" +
            "    float vConf = smoothstep(0.05, 0.35, conf);\n" +
            "    // Transition width scales with the gate (1.8x), so a strict\n" +
            "    // threshold also gets a steeper falloff instead of a wide\n" +
            "    // partially-trusted band (0.22*1.8 ~= the original fixed +0.18).\n" +
            "    float vRes  = 1.0 - smoothstep(uCutThresh, uCutThresh * 1.8, res);\n" +
            "    float vDiv  = 1.0 - smoothstep(uDivThresh, uDivThresh * 1.8, div);\n" +
            "    float valid = vConf * vRes * vDiv * (1.0 - outside);\n" +
            "\n" +
            "    // Forward-backward consistency: at a consistent surface the\n" +
            "    // forward vector into curr and the backward vector out of curr\n" +
            "    // cancel. A mismatch means one side of the pair is hidden there\n" +
            "    // (occlusion) and any blend would place content twice.\n" +
            "    //\n" +
            "    // Occlusion-aware side weighting (A): the FBC test at the warped\n" +
            "    // position only says the pair is globally consistent there. A\n" +
            "    // second test at THIS pixel (forward mv vs backward mv at vUV)\n" +
            "    // localises the occlusion. At a consistent pixel the phase-\n" +
            "    // balanced mix is kept unchanged; at a suspected occlusion the\n" +
            "    // blend leans hard toward the temporally nearest real frame —\n" +
            "    // trading a tiny early snap for removing the double-image ghost.\n" +
            "    // With the backward field disabled (FBC off) ph stays exactly\n" +
            "    // uPhase, i.e. the previous behaviour.\n" +
            "    float ph = uPhase;\n" +
            "    if (uFbcValid > 0.5) {\n" +
            "        vec2 mvBwd = texture(uMvBwd, cc).rg;\n" +
            "        if (!(dot(mvBwd, mvBwd) < 1.0e10)) mvBwd = vec2(0.0);\n" +
            "        float fbc = length(mvPx + mvBwd);\n" +
            "        valid *= 1.0 - smoothstep(uFbcThresh, uFbcThresh * 2.0, fbc);\n" +
            "\n" +
            "        vec2 mvBwdHere = texture(uMvBwd, vUV).rg;\n" +
            "        float bHere = (dot(mvBwdHere, mvBwdHere) < 1.0e10) ? length(mvPx + mvBwdHere) : 1.0e6;\n" +
            "        float consHere = 1.0 - smoothstep(uFbcThresh, uFbcThresh * 2.0, bHere);\n" +
            "        ph = mix(uPhase, uPhase < 0.5 ? 0.0 : 1.0, 1.0 - consHere);\n" +
            "    }\n" +
            "\n" +
            "    vec4 synth = mix(cPrev, cCurr, ph);\n" +
            "    vec4 nearest = (uPhase < 0.5) ? cPrev : cCurr;\n" +
            "    vec4 outc = mix(nearest, synth, valid);\n" +
            "\n" +
            "    // Static-pixel / HUD protection: content identical in both real\n" +
            "    // frames is copied from curr without any motion. Codec noise\n" +
            "    // keeps this from firing on normal video, but game overlays,\n" +
            "    // crosshairs and scoreboards are rock-stable and must never be\n" +
            "    // displaced by an estimator guess.\n" +
            "    vec4 cDirect = texture(uCurr, vUV);\n" +
            "    vec4 pDirect = texture(uPrev, vUV);\n" +
            "    float sd = dot(abs(pDirect.rgb - cDirect.rgb), vec3(0.3333));\n" +
            "    float isStatic = 1.0 - smoothstep(0.008, 0.025, sd);\n" +
            "    outc = mix(outc, cDirect, isStatic);\n" +
            "\n" +
            "    // Binary validity in alpha for the inpaint pass (continuous\n" +
            "    // validity there would soften every partially-trusted pixel).\n" +
            "    float validBin = max(step(0.45, valid), isStatic);\n" +
            "    outColor = vec4(outc.rgb, validBin);\n" +
            "}\n";

    // Occlusion inpaint: input RGBA carries per-pixel validity in alpha.
    // Fully-invalid pixels (disocclusion holes, frame borders) are filled
    // with a validity-weighted average of their 8 neighbours at a 2-texel
    // radius — enough to bridge typical 1..4px disocclusion seams without a
    // full multi-pass diffusion. Written branchless (all texture fetches in
    // uniform control flow) so it is safe on every mobile GPU driver.
    private static final String INPAINT_FRAG =
            "#version 300 es\n" +
            "precision highp float;\n" +
            "uniform mediump sampler2D uSrc;\n" +
            "uniform vec2 uTexel;\n" +
            "in vec2 vUV;\n" +
            "out vec4 outColor;\n" +
            "\n" +
            "void main() {\n" +
            "    vec4 c = texture(uSrc, vUV);\n" +
            "    vec3 acc = vec3(0.0);\n" +
            "    float wsum = 0.0;\n" +
            "    for (int y = -1; y <= 1; y++) {\n" +
            "        for (int x = -1; x <= 1; x++) {\n" +
            "            if (x == 0 && y == 0) continue;\n" +
            "            vec2 uv = clamp(vUV + vec2(float(x), float(y)) * uTexel * 2.0,\n" +
            "                            vec2(0.0), vec2(1.0));\n" +
            "            vec4 n = texture(uSrc, uv);\n" +
            "            float w = n.a * n.a;\n" +
            "            acc += n.rgb * w;\n" +
            "            wsum += w;\n" +
            "        }\n" +
            "    }\n" +
            "    vec3 filled = (wsum > 0.0001) ? acc / wsum : c.rgb;\n" +
            "    outColor = vec4(mix(c.rgb, filled, 1.0 - step(0.5, c.a)), 1.0);\n" +
            "}\n";

    // Quality telemetry (H): a tiny 16x16 additive-accumulation target.
    // Every synthetic frame adds one sparse sample per texel (256 points
    // over the frame); once per second the accumulated sums are read back
    // and logged as the invalid-pixel (disocclusion / gate-fail) rate, so
    // tuning can be judged on numbers instead of feel. Blending on RGBA16F
    // only needs EXT_color_buffer_float, already required for the motion
    // fields; if a driver refuses the blend the metric merely degrades to
    // a last-frame sample and nothing breaks.
    private static final String METRICS_FRAG =
            "#version 300 es\n" +
            "precision mediump float;\n" +
            "uniform mediump sampler2D uSrc;\n" +
            "in vec2 vUV;\n" +
            "out vec4 outColor;\n" +
            "void main() {\n" +
            "    vec4 c = texture(uSrc, vUV);\n" +
            "    outColor = vec4(1.0 - c.a, 1.0, 0.0, 0.0);\n" +
            "}\n";

    // ─── Imports that need no ambiguity ─────────────────────────────────────
    private static final String TAG = "FrameInterp";
}
