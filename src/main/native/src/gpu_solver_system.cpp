#include "gpu_solver_system.h"


namespace solver {

GPUSolverSystem::GPUSolverSystem(cudaStream_t stream) : stream(stream) {}


/// <summary>
/// Set this iteration solver mode
/// </summary>
/// <param name="solverMode"></param>
void GPUSolverSystem::setSolverMode(SolverMode solverMode) {
    this->solverMode = solverMode;

    switch (solverMode) {
    case SolverMode::DENSE_LU:
        if (!solverDenseLU)
            solverDenseLU = std::make_unique<solver::GPUSolverDenseLU>(stream);
        break;
    case SolverMode::SPARSE_QR:
        if (!solverSparseQR)
            solverSparseQR = std::make_unique<solver::GPUSolverSparseQR>(stream);
        break;
    case SolverMode::SPARSE_ILU:
        if (!solverSparseILU02)
            solverSparseILU02 = std::make_unique<solver::GPUSolverSparseILU02>(stream);
        break;
    }
}

/// <summary>
/// Solve Danse Matrix A system of linear equations.
/// </summary>
void GPUSolverSystem::solveSystemDN(double *A, double *b, int N) { 
    ///
    solverDenseLU->solveSystem(A, b, N);
}

/// <summary>
/// Solve Sparse Matrix A system of linear equations.
///
/// Matrix A - csr format
/// </summary>
/// <param name="csrRowInd"></param>
/// <param name="csrColInd"></param>
/// <param name="csrValInd"></param>
/// <param name="b"></param>
/// <param name="x"></param>
void GPUSolverSystem::solveSystemSP(int m, int n, int nnz, int *csrRowPtr, int *csrColInd, double *csrVal, double *b, double *x, int *singularity) {
    switch (solverMode) {
    case SolverMode::SPARSE_QR:
        solverSparseQR->solveSystem(m, n, nnz, csrRowPtr, csrColInd, csrVal, b, x, singularity);
        break;
    case SolverMode::SPARSE_ILU:
        solverSparseILU02->solveSystem(m, n, nnz, csrRowPtr, csrColInd, csrVal, b, x, singularity);
        break;
    default:
        fprintf(stderr, "[gpu.solver] unrecognized solver mode error !\n");
    }
}


} // namespace solver

#undef validateStream
