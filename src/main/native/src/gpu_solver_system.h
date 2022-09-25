#ifndef _SOLVER_SYSTEM_H_
#define _SOLVER_SYSTEM_H_

#include "cuda_runtime_api.h" /// Cuda Runtime API

/// solver implementations
#include "solver/gpu_solver_dense_lu.h"
#include "solver/gpu_solver_sparse_qr.h"
#include "solver/gpu_sparse_precondition_ilu02.h"

#include <memory>

namespace solver {


class GPUSolverSystem {
  public:
    /// <summary>
    /// 
    /// </summary>
    /// <param name="stream"></param>
    GPUSolverSystem(cudaStream_t stream);


    /// <summary>
    /// Set this iteration solver mode
    /// </summary>
    /// <param name="solverMode"></param>
    void setSolverMode(SolverMode solverMode);


    /// <summary>
    /// Solve Danse Matrix A system of linear equations.
    /// </summary>
    void solveSystemDN(double *A, double *b, int N);


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
    void solveSystemSP(int m, int n, int nnz, int *csrRowPtr, int *csrColInd, double *csrVal, double *b, double *x, int *singularity);

    /// 
    ~GPUSolverSystem() = default;

  private:

  private:
    cudaStream_t stream;

    SolverMode solverMode;
       
    std::unique_ptr<solver::GPUSolverDenseLU> solverDenseLU;
    std::unique_ptr<solver::GPUSolverSparseQR> solverSparseQR;    
};

} // namespace solver

#endif //_SOLVER_SYSTEM_H_