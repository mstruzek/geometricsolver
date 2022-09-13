#ifndef _LINEAR_SYSTEM_CUH_
#define _LINEAR_SYSTEM_CUH_


#include "cuda_runtime_api.h"       /// Cuda Runtime API
#include "cublas_v2.h"              /// cuBlas

#include "cusparse.h"
#include "cusolverDn.h"             /// cuSolverDN - Cuda Solver Danse - infrastructure
#include "cusolverSp.h"             /// cuSolverSP - Cuda Solver Sparse - infrastructure


namespace solver {

class GPULinearSystem {
  public:
    GPULinearSystem(cudaStream_t stream);

    /// <summary>
    /// Execution plan for computation of system of linear equations- A * x  = B .
    /// A is a dense tensor !
    ///
    /// LU solver
    /// - cusolverDnDgetrf_bufferSize - evaluate neede computation buffer
    /// - cusolverDnDgetrf  - LU factorization routine
    /// - cusolverDnDgetrs  - LU solver routine
    /// </summary>
    /// <param name="A"></param>
    /// <param name="b"></param>
    /// <param name="N"></param>
    void solveLinearEquation(double *A, double *b, int N);

    /// <summary>
    /// Execution plan form computation of A *x = b equation .
    /// A is a Sparse Tensor. Sparse Tensor.
    ///
    /// QR solver with reorder routine "symrcm".
    /// 
    /// </summary>
    /// <param name="csrRowInd"></param>
    /// <param name="csrColInd"></param>
    /// <param name="csrValInd"></param>
    /// <param name="b"></param>
    /// <param name="x"></param>
    void solverLinearEquationSP(int m, int n, int nnz, int *csrRowInd, int *csrColInd, double *csrValues, double *b,
                                double *x, int *singularity);

    /// dtor
    ~GPULinearSystem();

  private:
    /// <summary>
    /// Debug current stream execution state !
    /// </summary>
    void validateStreamState();

  private:
    cudaStream_t stream;

    /// cuda variables
    cusolverDnHandle_t handle;

    cusolverSpHandle_t cusolverSpHandle;

    cudaError_t lastError;

    /// cusolver context
    int Lwork = 0;

    /// additional workspace requirment imposed by LU solver
    double *Workspace = nullptr;

    /// lu factorization output vector pivot indices
    int *devIpiv = nullptr;

    /// data from Factorization or Solver
    int *devInfo = nullptr;

    /// cusolver tensor A default descriptor
    cusparseMatDescr_t descrA = nullptr;
};


} // namespace solver

#endif //_LINEAR_SYSTEM_CUH_