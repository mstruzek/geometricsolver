package com.mstruzek.jni;

public enum JNIDebugParameters {

    /**
     * stdout computed Tensor A = false
     */
    DEBUG_TENSOR_A(1),

    /**
     * stdout computed Tensor B = false
     */
    DEBUG_TENSOR_B(2),

    /**
     * additional debug messages from solver routine
     */
    DEBUG(3),

    /**
     * stdout State Vector = false
     */
    DEBUG_TENSOR_SV(4),

    /**
     * millisecond time  granularity
     */
    CLOCK_MILLISECONDS(5),

    /**
     * nanosecond time granularity
     */
    CLOCK_NANOSECONDS(6),

    /**
     * stdout constraint norm2
     */
    DEBUG_SOLVER_CONVERGENCE(8),

    /**
     * observe stream computations errors - e.g. MMU Fault - memory access failure or other...
     */
    DEBUG_CHECK_ARG(9),

    /**
     * DENSE_LAYOUT or SPARSE_LAYOUT
     * @class com.mstruzek.jni.JNIDebugCode.Computation
     */
    COMPUTATION_MODE(20),

    /**
     * LU, QR, ...
     */
    SOLVER_MODE(21),

    /**
     * compute Tensor A with Hessian - second derivatives
     */
    SOLVER_INC_HESSIAN(24),

    /**
     * This is  workspace size multiplier factor.( the main reason is for less re-allocations )
     */
    SOLVER_LWORK_FACTOR(26),

    /**
     * solution Epsilon : 10e-5
     */
    SOLVER_EPSILON(27),

    /**
     *  stdout computed CSR format
     */
    DEBUG_CSR_FORMAT(30),

    /**
     * stdout computed intermediate COO format
     */
    DEBUG_COO_FORMAT (31),

    /**
     * Stream Capturing Capabilities
     */
    STREAM_CAPTURING(60);

    public final int code;

    public void setBooleanProperty(boolean value) {
        JNISolverGate.setBooleanProperty(this.code, value);
    }

    public void setDoubleProperty(double value) {
        JNISolverGate.setDoubleProperty(this.code, value);
    }

    public void setLongProperty(long value) {
        JNISolverGate.setLongProperty(this.code, value);
    }

    public static class ComputationMode {
        /**
         * Dense matrix mode.
         */
        public static final int DENSE_MODE = 1;

        /**
         * Sparse matrix mode.
         */
        public static final int SPARSE_MODE = 2;

        /**
         * Mixed direct matrix mode.
         */
        public static final int DIRECT_MODE = 3;

        /**
         * Journal COO format that is post-processed into canonical CSR form.
         */
        public static final int COMPACT_MODE = 4;
    }

    public static class SolverMode {

        /**
         * dense LU factorization with LU solver - cusolverDnDgetrf
         */
        public static final int LU_DENSE = 1;

        /**
         * default QR factorization with solver for sparse matrix - cusolverSpDcsrlsvqr
         */
        public static final int QR_SPARSE = 2;

        /**
         *  Bi-Conjugate Gradient stabilized method with Incomplete LU precondition
        */
        public static final int ILU_BiCG_STAB = 3;
    }

    JNIDebugParameters(int code) {
        this.code = code;
    }

}
