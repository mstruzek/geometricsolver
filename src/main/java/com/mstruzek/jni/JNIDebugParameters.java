package com.mstruzek.jni;

public enum JNIDebugParameters {


    /**
     * additional debug messages from solver routine
     */
    DEBUG(0),

    /**
     * production kernel = true
     *
     * if production kernel thread-Id addressing is horizontal in single dispatch kernel,
     *
     * otherwise in order computation on the  single stream.
     */
    KERNEL_PRE(1),

    /**
     * stdout computed Tensor A = false
     */
    DEBUG_TENSOR_A(2),

    /**
     * stdout computed Tensor B = false
     */
    DEBUG_TENSOR_B(3),

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
     * compute Tensor A with Hessian - second derivatives
     */
    SOLVER_INC_HESSIAN(7),

    /**
     * stdout constraint norm2
     */
    DEBUG_SOLVER_CONVERGENCE(8),

    /**
     * observe stream computations errors - e.g. MMU Fault - memory access failure or other...
     */
    DEBUG_CHECK_ARG(9),

    /**
     * Set default computation kernel  Grid Size = 1
     */
    GRID_SIZE(10),
    /**
     * Set default computation kernel Block Size = 128
     */
    BLOCK_SIZE(11),

    /**
     * DENSE_LAYOUT or SPARSE_LAYOUT
     * @class com.mstruzek.jni.JNIDebugCode.Computation
     */
    COMPUTATION_MODE(12),

    /**
     * This is  workspace size multiplier factor.( the main reason is for less re-allocations )
     */
    CU_SOLVER_LWORK_FACTOR(21),

    /**
     * solution Epsilon : 10e-5
     */
    CU_SOLVER_EPSILON(22),

    /**
     * Stream Capturing Capabilities
     */
    STREAM_CAPTURING(100);

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

    public static class Computation {
        public static final int DENSE_LAYOUT  = 1;
        public static final int SPARSE_LAYOUT = 2;
        /// public static final int DIRECT_LAYOUT = 3; // *this is continue next steps operation from sparse layout !
    }


    JNIDebugParameters(int code) {
        this.code = code;
    }

}
