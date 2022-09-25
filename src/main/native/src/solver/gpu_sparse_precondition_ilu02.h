#ifndef _GPU_SOLVER_SPARSE_IL02_H_
#define _GPU_SOLVER_SPARSE_IL02_H_

#include "cuda_runtime_api.h"

#include "../quda.h"

namespace solver {

/// <summary>
/// Solver is configured wiht numerBoost routine so it is possible to change the cut-off for numeric zero value with some tolerance.
/// --- Preconditioner into BiCGSTAB ---
/// 
/// GPUSparsePreconditionILU02 {
/// 
/// }
/// </summary>
class GPUSparsePreconditionILU02 {

  public:
    /// <summary>
    /// 
    /// </summary>
    /// <param name="stream"></param>
    GPUSparsePreconditionILU02(cudaStream_t stream);

    /// <summary>
    /// 
    /// </summary>
    /// <param name="parameterId">[in] parameter specific confuguration id</param>
    /// <param name="valueInt">[in]</param>
    /// <param name="valueDouble">[in]</param>
    void configure(int parameterId, int valueInt, double valueDouble);

    /// <summary>
    /// Tensor Incomplete LU factorization 
    /// M = L * U
    /// CSR(csrRowPtr, csrColInd, csrVal) 
    /// Incomplete-LU sparse factorization.
    ///
    ///  A * x = L * U * x = b;
    ///
    /// 0. Incomplete-LU factorization
    ///   A ~~ L*U = M 
    /// 
    /// </summary>
    /// <param name="m">[in]</param>
    /// <param name="n">[in]</param>
    /// <param name="nnz">[in]</param>
    /// <param name="csrRowPtr">[in.out]</param>
    /// <param name="csrColInd">[in.out]</param>
    /// <param name="csrVal">[in.out]</param>
    /// <param name="b">[in]</param>
    /// <param name="x">[out]</param>
    /// <param name="singularity"></param>
    void tensorFactorization(int m, int n, int nnz, int *csrRowPtr, int *csrColInd, double *csrVal, double *b, double *x, int *singularity);

    /// <summary>
    /// Execution plan for system of linear qeuations A * x = b
    /// 
    /// 1. TRM solve        -- this is the routine !
    ///  L * z = b
    /// 
    /// 2. TRM solve
    ///  U * x = z
    /// 
    /// Routine reference: https://docs.nvidia.com/cuda/cusparse/index.html#csrilu02_solve
    /// </summary>
    /// <param name="m">[in] leading dimension</param>
    /// <param name="n">[in] number of columns</param>
    /// <param name="nnz">[in] non-zero elements in csr format</param>
    /// <param name="csrRowPtr">[in] csr device vector</param>
    /// <param name="csrColInd">[in] csr device vector</param>
    /// <param name="csrVal">[in] csr device vector</param>
    /// <param name="b">[in] right hand side vector</param>
    /// <param name="x">[out] solution dense vector </param>
    /// <param name="singularity">[out] - -1 invertible , otherwise j > 0 , index of diagonal element</param>
    void solveSystem(int m, int n, int nnz, int *d_csrRowPtr, int *csrColInd, double *csrVal, double *b, double *x, int *singularity);

    ~GPUSparsePreconditionILU02();

  private:

    /// <summary>
    /// setup matrix M , L , U descriptor and configure defaults
    /// </summary>
    void setupSparseMatDescriptors();

    /// <summary>
    /// initialize Incomplete-LU solver data info structures.
    /// </summary>
    void setupSolverInfoStructures();

    /// <summary>
    /// Request and set minimum buffer required for those routines.
    /// </summary>
    void requestEnsurePBufferSize(int m, int n, int nnz, int *d_csrRowPtr, int *d_csrColInd, double *d_csrVal);

    /// <summary>
    /// Perform Incomplete LU Analyssis on M, perfomr traingular solver analysis on Upper(U) and Lower(L) matrix.
    /// </summary>
    void performAnalysisIncompleteLU(int m, int n, int nnz, int *d_csrRowPtr, int *d_csrColInd, double *d_csrVal, int *singularity);
   

    /// <summary>
    /// debug utility - test kernel submission into stream , and synchronize on success computation state
    /// </summary>
    void validateStreamState();

  private:

    /// computation stream that the handler will commit work into.
    cudaStream_t stream;

    /// computation handler
    cusparseHandle_t handle;

    /// intermediate dense vector in TRSLV
    utility::dev_vector<double> d_z; 

    /// LU tensor descriptors
    cusparseMatDescr_t descr_M = NULL;
    cusparseMatDescr_t descr_L = NULL;
    cusparseMatDescr_t descr_U = NULL;
    
    /// Incomplete-LU factorization ( possibly computed in first iteration )
    csrilu02Info_t info_M = NULL;

    /// Lower: sparse triangular linear system solver [DEPRECATED]  ( possibly computed in first iteration )
    csrsv2Info_t info_L = NULL;

    /// Upper: sparse triangular linear system solver [DEPRECATED] ( possibly computed in first iteration )
    csrsv2Info_t info_U = NULL;
    
    int pBufferSize_M;
    int pBufferSize_L;
    int pBufferSize_U;
    int pBufferSize;
    
    /// shared single stream buffer M,L,U
    utility::dev_vector<char> pBuffer;
    
    /// index of structural diagonal element from ILU analysis - singularity
    int structural_zero;

    /// infex of structural zeror element form ILU factorization - singualrity
    int num_zero;
    
    /// crvsv2 - factor 
    const double alpha = 1.;
    
    const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_NO_LEVEL; /// ??? "to disable level information"
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL; /// CUSPARSE_SOLVE_POLICY_NO_LEVEL
    
    const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL; /// more parallelism for non-unitary diagonal ( structura_zero )

    /// do not transpose
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseOperation_t trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
};

} // namespace solver

#endif // !_GPU_SOLVER_SPARSE_IL02_H_