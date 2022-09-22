#ifndef _GPU_SOLVER_SPARSE_BICGSTAB_H_
#define _GPU_SOLVER_SPARSE_BICGSTAB_H_

#include "cuda_runtime_api.h"

#include "cusolverSp.h"

namespace solver {

/// <summary>
/// Unpreconditioned BiCGSTAB  /// https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
///     https://docs.nvidia.com/cuda/incomplete-lu-cholesky/index.html
/// </summary>
class GPUSolverSparseBiCGSTAB {
  public:
    /// <summary>
    /// Seutp solver in Sparse mode, associated provided stream into computation context.
    /// </summary>
    /// <param name="stream"></param>
    GPUSolverSparseBiCGSTAB(cudaStream_t stream);

    /// <summary>
    /// Solver configuration parameters.
    /// </summary>
    /// <param name="parameterId"></param>
    /// <param name="valueInt"></param>
    /// <param name="valueDouble"></param>
    void configure(int parameterId, int valueInt, double valueDouble);

    /// <summary>
    /// Execution plan form system of linear equations  A *x = b .
    /// A is a Sparse Tensor. Sparse Tensor.
    ///
    /// QR solver with reorder routine "symrcm".
    /// </summary>
    /// <param name="m">[in]</param>
    /// <param name="n">[in]</param>
    /// <param name="nnz">[in] non-zero elements</param>
    /// <param name="csrRowPtrA">[in]</param>
    /// <param name="csrColIndA">[in]</param>
    /// <param name="csrValA">[in]</param>
    /// <param name="b">[in]</param>
    /// <param name="x">[out]</param>
    /// <param name="singularity">[out]</param>
    void solveSystem(int m, int n, int nnz, int *csrRowPtrA, int *csrColIndA, double *csrValA, double *b, double *x, int *singularity);
       

    ~GPUSolverSparseBiCGSTAB();

  private:
    /// <summary>
    /// CSR tensor
    /// </summary>
    void setupTensorADescriptor();

    /// <summary>
    /// Debug mode utility.
    /// test strem state after kernel submission and after completion
    /// </summary>
    void validateStreamState();

  private:
    /// computation stream that the handler will commit work into.
    cudaStream_t stream;

    cusolverSpHandle_t handle;

    /// cusolver tensor A default descriptor
    cusparseMatDescr_t descrA = nullptr;

    /// Tolerance to decide if singular or not .
    double tolerance = 1e-10;

};

} // namespace solver



#endif // !_GPU_SOLVER_SPARSE_BICGSTAB_H_
