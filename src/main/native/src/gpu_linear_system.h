#ifndef _LINEAR_SYSTEM_CUH_
#define _LINEAR_SYSTEM_CUH_


#include "cuda_runtime_api.h"       /// Cuda Runtime API
#include "cublas_v2.h"              /// cuBlas
#include "cusolverDn.h"             /// cuSolverDN - Cuda Solver Danse - infrastructure


namespace solver {

class GPULinearSystem {
  public:

    GPULinearSystem(cudaStream_t stream);

    /// <summary>
    /// Execution plan for computation of linear system - A * x  = B
    /// LU solver 
    /// - cusolverDnDgetrf_bufferSize - evaluate neede computation buffer 
    /// - cusolverDnDgetrf  - LU factorization routine 
    /// - cusolverDnDgetrs  - LU solver routine
    /// </summary>
    /// <param name="A"></param>
    /// <param name="b"></param>
    /// <param name="N"></param>
    void solveLinearEquation(double *A, double *b, size_t N);


    /// <summary>
    /// compute standard Euclidean norm - cublasDnrm2
    /// </summary>
    /// <param name="n"></param>
    /// <param name="x"></param>
    /// <param name="result"></param>
    void vectorNorm(int n, double *x, double *result);

    ~GPULinearSystem();

  private:
    cudaStream_t _stream = nullptr;

    /// cuda variables
    cusolverDnHandle_t handle = nullptr;

    cublasHandle_t cublasHandle = nullptr;

    cudaError_t lastError = cudaSuccess;

    /// cusolver context
    int Lwork = 0;

    /// additional workspace requirment imposed by LU solver
    double *Workspace = nullptr;

    /// lu factorization output vector pivot indices
    int *devIpiv = nullptr;

    /// data from Factorization or Solver
    int *devInfo = nullptr;
};

} // namespace solver

#endif //_LINEAR_SYSTEM_CUH_