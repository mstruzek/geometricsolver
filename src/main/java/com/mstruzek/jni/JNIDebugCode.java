package com.mstruzek.jni;

public enum JNIDebugCode {
    DEBUG(0),
    DEBUG_TENSOR_A(2),
    DEBUG_TENSOR_B(3),
    DEBUG_TENSOR_SV(4),
    CLOCK_MILLISECONDS(5),
    CLOCK_NANOSECONDS(6),
    SOLVER_INC_HESSIAN(7),
    DEBUG_SOLVER_CONVERGENCE(8),
    DEBUG_CHECK_ARG(9),
    GRID_SIZE(10),
    BLOCK_SIZE(11),
    CU_SOLVER_LWORK_FACTOR(21);

    public final int code;

    JNIDebugCode(int code) {
        this.code = code;
    }
    }
