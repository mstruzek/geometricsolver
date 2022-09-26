#ifndef _GPU_SOLVER_SPARSE_BICGSTAB_H_
#define _GPU_SOLVER_SPARSE_BICGSTAB_H_

#include "cuda_runtime_api.h"

#include "cublas_v2.h"
#include "cusparse.h"

#include "../quda.h"

#include "gpu_sparse_precondition_ilu02.h"


namespace solver {

/// <summary>
/// Unpreconditioned Biconjugate Gradient Stabilized  method - BiCGStab 
/// https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
/// https://docs.nvidia.com/cuda/incomplete-lu-cholesky/index.html
/// 
/// https://github.com/tpn/cuda-samples/blob/master/v10.2/7_CUDALibraries/BiCGStab/pbicgstab.cpp
/// 
///  #? e:\gpgpu\cuda-samples\Samples\4_CUDA_Libraries\conjugateGradientCudaGraphs\conjugateGradientCudaGraphs.cu
/// 
/// In numerical linear algebra, the biconjugate gradient stabilized method, often abbreviated as BiCGSTAB, 
/// is an iterative method developed by H. A. van der Vorst for the numerical solution 
/// of nonsymmetric linear systems.
/// </summary>
class GPUSolverSparseBiCGSTAB {
  public:
    /// <summary>
    /// Seutp solver in Sparse mode, associated provided stream into computation context.
    /// </summary>
    /// <param name="stream"></param>
    GPUSolverSparseBiCGSTAB(cudaStream_t stream);

    GPUSolverSparseBiCGSTAB(const GPUSolverSparseBiCGSTAB &) = delete;

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
    /// Allocate temporary buffers for at least m - rows
    /// </summary>
    /// <param name="m"></param>
    void allocateEvaluationBuffers(size_t m);

    /// <summary>
    /// Deallocate temporary buffers.
    /// </summary>
    void deallocateTempBuffers();

    /// <summary>
    /// 
    /// Debug mode utility.
    /// test strem state after kernel submission and after completion
    /// </summary>
    void validateStreamState();

  private:
    /// computation stream that the handler will commit work into.
    cudaStream_t stream;

    /// CuSparse Handle corellated with prior stream
    cusparseHandle_t handle;

    /// cuBlas Handle corellated with prior stream
    cublasHandle_t cublasHandle;

    /// cusolver tensor A default descriptor
    cusparseMatDescr_t descrA = NULL;

    /// Incomplete LU prcondition solver
    std::unique_ptr<solver::GPUSparsePreconditionILU02> sparsePrconditionILU;

    /// additional destructible vectors for sparse precondition method
    utility::dev_vector<int> d_csrRowPtrM;
    utility::dev_vector<int> d_csrColIndM;
    utility::dev_vector<double> d_csrValM;

    /// Tolerance to decide if singular or not .
    double tolerance = 10e-20;

    /// -------------- SpMV details -----------------
    ///         Y = alfa * op(A) *X + beta* Y;
    ///

    /// input csr tensor definition from SpMV descriptor
    cusparseSpMatDescr_t matA = NULL;

    /// input vector from SpMV
    cusparseDnVecDescr_t vecX = NULL;

    /// output vector from SpMV
    cusparseDnVecDescr_t vecY = NULL;

    cudaDataType computeType = CUDA_R_64F;

    ///  SpMV
    size_t bufferSize;

    /// allocated buffer for SpMV
    void *externalBuffer = NULL;

    /// computation temporary vectors
    int alloc_m;
    double *x0 = NULL, *xi = NULL;             /// default to x0 = cublasDcopy(handle, n, xi, 1, x, 1); // nie konicznie , referencja wystarczy !
    double *r0 = NULL, *ri = NULL, *rp = NULL; /// r pod daszkiem
    double *v0 = NULL, *vi = NULL, *p0 = NULL, *pi = NULL, *ph = NULL;           /// p i p^

    double *s = NULL, *sh = NULL; /// s and s^
    double *t = NULL;            ///
};

} // namespace solver

#endif // !_GPU_SOLVER_SPARSE_BICGSTAB_H_
