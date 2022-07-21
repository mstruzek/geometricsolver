package com.mstruzek.msketch.solver.jni;

import java.nio.ByteBuffer;

/**
 */
public class JniNativeSolver {

    /*
     * Requirements:
     * - single interface into JNI C20 without necessary recompilation of header file even in case of new methods or objects.
     * - additionally all infrastructure for serialization/deserialization into ByteBuffers.
     * - optionally exist more handy solution like SWIG, for wider integration with C++.
     */

    /**
     * Register new or resize actual shared buffer for internal communication.
     * @param capacity byte buffer capacity
     * @return address ? UNSAFE set address
     */
    static native long malloc(long capacity);

    /**
     * Free allocated memory referenced by address.
     * @param address
     */
    static native void free(long address);

    /**
     * Enclose block of memory into DirectByteBuffer.
     * @param address
     * @param capacity
     * @return DirectByteBuffer UNSAFE.getFild("address")
     */
    static native ByteBuffer wrapBuffer(long address, long capacity);

    /**
     * Serialize object into shared buffer in Java, and deserializer on the other side.
     */
    static native void readObject(long address, long type, int id);

    /**
     * Serialize object into shared buffer, and deserialize in Java.
     */
    static native void writeObject(long address, long type, int id);

    /**
     * Run single execution function identified by Id, usually kernel function or cublas api function.
     * @return 0 on success otherwise read error status.
     */
    static native int execute(int action);

    /**
     * Error should be registered for system invalid actions, failures of  cuda API, failures of  CUBLAS invocation ?.
     * @return 1: system error, 2: GPU API, 4: CUBLAS API
     */
    static native int errorType();

    /**
     * Get last registered error code from application or kernel space.
     * @return error code , depends on source space.
     */
    static native int errorCode();

    /**
     * Get last error byte buffer message. Byte array initialized in JVM arena.
     * @return UTF-8 encoded string ( ,? UTF-16 )
     */
    static native byte[] errorMessage();

}
