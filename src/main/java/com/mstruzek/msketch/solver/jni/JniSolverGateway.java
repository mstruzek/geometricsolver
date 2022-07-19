package com.mstruzek.msketch.solver.jni;

/**
 *  nie wygdonie tak malym interfejsem -- albo SWIG !
 *
 *  w takim modelu zydkujemy serializacje/deserializacje w javie i w C20  !
 *
 *  - mniej rekompilacji tej wersji naglowka !
 *
 *  - wspolny kontrakt nierozerwalny , single version !
 *
 */
public class JniSolverGateway {

    /**
     * Register new or resize actual shared buffer for internal communication.
     * @param size byte buffer capacity
     * @return address ? UNSAFE set address
     */
    native long registerBuffer(long size);

    native long freeBuffer(long address);

    /**
     * Serialize object into shared buffer in Java, and deserializer on the other side.
     */
    native void readObject(long type, int id);

    /**
     * Serialize object into shared buffer, and deserialize in Java.
     */
    native void writeObject(long type, int id);

    /**
     * Run single execution function identified by Id, usually kernel function or cublas api function.
     * @return 0 on success otherwise read error status.
     */
    native int execute(int action);

    /**
     * Error should be registered for system invalid actions, failures of  cuda API, failures of  CUBLAS invocation ?.
     * @return 1: system error, 2: GPU API, 4: CUBLAS API
     */
    native int errorType();

    /**
     * Get last registered error code from application or kernel space.
     * @return error code , depends on source space.
     */
    native int errorCode();

    /**
     * Get last error byte buffer message. Byte array initialized in JVM arena.
     * @return UTF-8 encoded string ( ,? UTF-16 )
     */
    native byte[] errorMessage();

}
