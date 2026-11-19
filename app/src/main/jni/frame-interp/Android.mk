LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := frame-interp

LOCAL_SRC_FILES := frame_interp_jni.c

LOCAL_CFLAGS += -O3 -Wall

LOCAL_LDLIBS := -llog -lEGL -lGLESv3

include $(BUILD_SHARED_LIBRARY)
