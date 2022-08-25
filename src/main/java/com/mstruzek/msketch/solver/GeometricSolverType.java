package com.mstruzek.msketch.solver;

public enum GeometricSolverType {

    /**
     * Default CPU Solver with Colt integration for LU factorization/solver
     *
     *
     */
    CPU_SOLVER,


    /**
     * Default GPU Solver with Nvidia CUDA Runtime integration:
     *
     * [ Cuda Runtime  Api ]    - encode/decode input data for `A `b tensors
     * [ cuSOLVER ]             - LU factorization/solver
     * [ cuBLAS ]               - kernels for error computation
     */
    GPU_SOLVER;

}
