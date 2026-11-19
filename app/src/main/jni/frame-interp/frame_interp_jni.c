/*
 * JNI shim exposing Qualcomm Adreno frame-generation GPU extensions
 * (GL_QCOM_motion_estimation / GL_QCOM_texture_extrapolation) to the
 * FrameInterpolationRenderer on the Java side.
 *
 * The vendor entry points are not part of the Android SDK/GLES headers and
 * must be resolved at runtime via eglGetProcAddress. All functions here must
 * be invoked with the rendering EGLContext current (i.e. from the GL thread).
 *
 * Extension call conventions follow IRedDragonICY/afme (Apache-2.0):
 *   glTexEstimateMotionQCOM(prevLuma, currLuma, mvOut)
 *     - single-channel (R8) luma textures, block-resolution RGBA16F MV output,
 *       .xy = pos_curr - pos_prev in pixels
 *   glExtrapolateTex2DQCOM(prev, curr, out, factor)
 *     - RGBA8 frame textures, driver-internal extrapolator
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include <jni.h>
#include <string.h>
#include <stdlib.h>
#include <android/log.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <GLES3/gl3ext.h>
#include <GLES2/gl2ext.h>

#ifndef GL_MOTION_ESTIMATION_SEARCH_BLOCK_X_QCOM
#define GL_MOTION_ESTIMATION_SEARCH_BLOCK_X_QCOM 0x8C90
#endif
#ifndef GL_MOTION_ESTIMATION_SEARCH_BLOCK_Y_QCOM
#define GL_MOTION_ESTIMATION_SEARCH_BLOCK_Y_QCOM 0x8C91
#endif

#define LOG_TAG "FrameInterp"
#define ALOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)

typedef void (*PFNGLTEXESTIMATEMOTIONQCOMPROC)(GLuint, GLuint, GLuint);
typedef void (*PFNGLEXTRAPOLATETEX2DQCOMPROC)(GLuint, GLuint, GLuint, GLfloat);

static PFNGLTEXESTIMATEMOTIONQCOMPROC sEstimateMotion = NULL;
static PFNGLEXTRAPOLATETEX2DQCOMPROC sExtrapolateTex2D = NULL;
static int sResolved = 0;

static void resolveExtensions(void) {
    if (sResolved) {
        return;
    }
    sResolved = 1;
    sEstimateMotion = (PFNGLTEXESTIMATEMOTIONQCOMPROC) eglGetProcAddress("glTexEstimateMotionQCOM");
    sExtrapolateTex2D = (PFNGLEXTRAPOLATETEX2DQCOMPROC) eglGetProcAddress("glExtrapolateTex2DQCOM");
}

/*
 * Returns a bitmask of supported QCOM features:
 *   bit 0 (1) = GL_QCOM_motion_estimation usable
 *   bit 1 (2) = GL_QCOM_texture_extrapolation usable
 */
JNIEXPORT jint JNICALL
Java_com_limelight_utils_FrameInterpolationRenderer_nativeQueryQcomFeatures(JNIEnv *env, jclass clazz) {
    resolveExtensions();

    const char *ext = (const char *) glGetString(GL_EXTENSIONS);
    if (ext == NULL) {
        return 0;
    }

    int features = 0;
    if (sEstimateMotion != NULL && strstr(ext, "GL_QCOM_motion_estimation") != NULL) {
        features |= 1;
    }
    if (sExtrapolateTex2D != NULL && strstr(ext, "GL_QCOM_texture_extrapolation") != NULL) {
        features |= 2;
    }
    return features;
}

/*
 * Runs the hardware motion estimator. Returns JNI_TRUE on success.
 * Must be called with prevLuma/currLuma bound-able R8 textures and an
 * RGBA16F mv texture at block resolution.
 */
JNIEXPORT jboolean JNICALL
Java_com_limelight_utils_FrameInterpolationRenderer_nativeEstimateMotion(JNIEnv *env, jclass clazz,
                                                                         jint prevLuma, jint currLuma, jint mvTex) {
    resolveExtensions();
    if (sEstimateMotion == NULL) {
        return JNI_FALSE;
    }
    while (glGetError() != GL_NO_ERROR) {}
    sEstimateMotion((GLuint) prevLuma, (GLuint) currLuma, (GLuint) mvTex);
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        ALOGW("glTexEstimateMotionQCOM failed, glError=0x%x", err);
        return JNI_FALSE;
    }
    return JNI_TRUE;
}

/*
 * Runs the driver's black-box frame extrapolator.
 * prev/curr/out are RGBA8 frame textures, factor in [0,1].
 */
JNIEXPORT jboolean JNICALL
Java_com_limelight_utils_FrameInterpolationRenderer_nativeExtrapolate(JNIEnv *env, jclass clazz,
                                                                      jint prevTex, jint currTex, jint outTex, jfloat factor) {
    resolveExtensions();
    if (sExtrapolateTex2D == NULL) {
        return JNI_FALSE;
    }
    while (glGetError() != GL_NO_ERROR) {}
    sExtrapolateTex2D((GLuint) prevTex, (GLuint) currTex, (GLuint) outTex, factor);
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        ALOGW("glExtrapolateTex2DQCOM failed, glError=0x%x", err);
        return JNI_FALSE;
    }
    return JNI_TRUE;
}

/*
 * Queries the hardware motion estimation block size configured by the driver
 * for the currently bound mv texture. Returns true if the values were
 * verified (no GL error), false if the enums are unsupported.
 */
JNIEXPORT jboolean JNICALL
Java_com_limelight_utils_FrameInterpolationRenderer_nativeQueryMotionBlock(JNIEnv *env, jclass clazz,
                                                                           jobject out) {
    resolveExtensions();
    if (sEstimateMotion == NULL) {
        return JNI_FALSE;
    }

    GLint blockX = 16, blockY = 16;
    while (glGetError() != GL_NO_ERROR) {}
    glGetIntegerv(GL_MOTION_ESTIMATION_SEARCH_BLOCK_X_QCOM, &blockX);
    glGetIntegerv(GL_MOTION_ESTIMATION_SEARCH_BLOCK_Y_QCOM, &blockY);
    if (glGetError() != GL_NO_ERROR) {
        return JNI_FALSE;
    }

    jclass cls = (*env)->GetObjectClass(env, out);
    jfieldID fx = (*env)->GetFieldID(env, cls, "x", "I");
    jfieldID fy = (*env)->GetFieldID(env, cls, "y", "I");
    (*env)->SetIntField(env, out, fx, (jint) blockX);
    (*env)->SetIntField(env, out, fy, (jint) blockY);
    return JNI_TRUE;
}
