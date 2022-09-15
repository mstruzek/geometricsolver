#ifndef _GPU_SOLVER_DANSE_LU_H_
#define _GPU_SOLVER_DANSE_LU_H_

#include "cuda_runtime_api.h"

#include "cusolverDn.h"

namespace solver {

class GPUSolverDenseLU {
  public:
    /// <summary>
    ///
    /// </summary>
    /// <param name="stream"></param>
    GPUSolverDenseLU(cudaStream_t stream);

    /// <summary>
    /// Execution plan for computation of system of linear equations- A * x  = B .
    /// A is a dense tensor !
    ///
    /// LU solver
    /// - cusolverDnDgetrf_bufferSize - evaluate neede computation buffer
    /// - cusolverDnDgetrf  - LU factorization routine
    /// - cusolverDnDgetrs  - LU solver routine
    /// </summary>
    /// <param name="A">[in]</param>
    /// <param name="b">[in/out]</param>
    /// <param name="N">leading dimension</param>
    void solveSystem(double *A, double *b, int N);

    ~GPUSolverDenseLU();

  private:
    cudaStream_t stream;

    /// cuda variables
    cusolverDnHandle_t handle;

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

#endif // _GPU_SOLVER_DANSE_LU_H_
