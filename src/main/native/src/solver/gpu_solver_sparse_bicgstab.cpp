#include "gpu_solver_sparse_bicgstab.h"

#include "../cuerror.h"
#include "../gpu_utility.h"
#include "../settings.h"

#ifdef DEBUG_GPU
#define validateStream validateStreamState()
#else
#define validateStream
#endif

#define HANDLE_STATUS status =
#define HANDLE_BLAS_STATUS cublasStatus =

namespace solver {

GPUSolverSparseBiCGSTAB::GPUSolverSparseBiCGSTAB(cudaStream_t stream) : stream(stream), alloc_m(0), bufferSize(0), d_csrRowPtrM(), d_csrColIndM(), d_csrValM() {
        cusparseStatus_t status;
        cublasStatus_t cublasStatus;

        sparsePrconditionILU = std::make_unique<solver::GPUSparsePreconditionILU02>(stream);

        /// cuSPARSE
        HANDLE_STATUS cusparseCreate(&handle);
        if (status) {
                fprintf(stderr, "[solver.error] handler failure;  %s  \n", cusparseGetErrorName(status));
                exit(-1);
        }
        HANDLE_STATUS cusparseSetStream(handle, stream);
        if (status) {
                fprintf(stderr, "[cusolver/error] set stream failure;  %s  \n", cusparseGetErrorName(status));
                exit(-1);
        }

        /// cuBLAS
        HANDLE_BLAS_STATUS cublasCreate(&cublasHandle);
        if (cublasStatus) {
                fprintf(stderr, "[cublas/error] cublas setup failed ; %s  %s\n", cublasGetStatusName(cublasStatus), cublasGetStatusString(cublasStatus));
                exit(-1);
        }
        HANDLE_BLAS_STATUS cublasSetStream(cublasHandle, stream);
        if (cublasStatus) {
                fprintf(stderr, "[cublas/error] cublas stream setup failed ; %s  %s\n", cublasGetStatusName(cublasStatus), cublasGetStatusString(cublasStatus));
                exit(-1);
        }
}

void GPUSolverSparseBiCGSTAB::configure(int parameterId, int valueInt, double valueDouble) {

#define PARAMETER_SOLVER_BiCGSTAB_PRECONDITIONER 1
#define PARAMETER_SOLVER_BiCGSTAB_TOLERANCE 2

        switch (parameterId) {
        case PARAMETER_SOLVER_BiCGSTAB_PRECONDITIONER:
                // default 0 , otherwise use reduce fill-in  schema
                break;
        case PARAMETER_SOLVER_BiCGSTAB_TOLERANCE:
                //#define SOLVER_QR_TOLERANCE 1e-10
                tolerance = valueDouble;
                break;
        default:
                return;
        }
}

#define CONVERGENCE_ROUND 40
#define HANDLE_STATUS status =

#define CSR(csrRowPtrA, csrColIndA, csrValA) csrRowPtrA, csrColIndA, csrValA

void GPUSolverSparseBiCGSTAB::solveSystem(int m, int n, int nnz, int *csrRowPtrA, int *csrColIndA, double *csrValA, double *b, double *x, int *singularity) {

    cublasStatus_t cublasStatus;
    cusparseStatus_t status;

        // clang-format off

/// 3. ro0 = alpha = omega0 = 1
    double ro0 = 1.0;           /// [i-1]
    double roi = 1.0;           /// []
    double omega0 = 1.0;        ///  [i-1]
    double omegai = 1.0;
    double alpha = 1.0;
    double beta = 0.0;

    /// real SpMV or cublasDdot multipliers !
    double alpha_r = 0.0;
    double beta_r = 0.0;

    /// iteration norm 2
    double nrm2 = 0.0;
    double nrmr = 1.0;
    double nrmr0 = 1.0;

    cudaEvent_t begin;
    cudaEvent_t finish;
    
    cudaEvent_t start[CONVERGENCE_ROUND] ={};
    cudaEvent_t stop[CONVERGENCE_ROUND] = {};

    checkCudaStatus(cudaEventCreate(&begin));
    checkCudaStatus(cudaEventCreate(&finish));
    for(int i = 0 ; i < CONVERGENCE_ROUND ; ++i) {
        checkCudaStatus(cudaEventCreate(&start[i]));
        checkCudaStatus(cudaEventCreate(&stop[i]));
    }

    checkCudaStatus(cudaEventRecord(begin, stream));
    
/// -2. allocate this evaluation vectors if necessary
    allocateEvaluationBuffers(m);

/// -1. setup new tensor descriptor
    if (matA) {
        checkCusparseStatus(cusparseDestroySpMat(matA));
        checkCusparseStatus(cusparseDestroyDnVec(vecX));
        checkCusparseStatus(cusparseDestroyDnVec(vecY));
    }
    /// ---
    /// `statyczna macierz A - SparseMV
    checkCusparseStatus(cusparseCreateCsr(&matA, m, n, nnz, CSR(csrRowPtrA, csrColIndA, csrValA), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, 
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    checkCusparseStatus(cusparseCreateDnVec(&vecX, m, x0, CUDA_R_64F));
    checkCusparseStatus(cusparseCreateDnVec(&vecY, m, r0, CUDA_R_64F));

/// precondition setup - tensor M in csr format
    d_csrRowPtrM = utility::dev_vector<int>(m + 1);         /// uzupelnic w takim modelu bytebuffer - capacity_ensure(size_t ) ?# dev_vector { .capacity = , .size = }
    d_csrColIndM = utility::dev_vector<int>(nnz);
    d_csrValM = utility::dev_vector<double>(nnz);

    d_csrRowPtrM.memcpy_of(utility::dev_vector<int>(csrRowPtrA, m + 1, stream));
    d_csrColIndM.memcpy_of(utility::dev_vector<int>(csrColIndA, nnz, stream));
    d_csrValM.memcpy_of(utility::dev_vector<double>(csrValA, nnz, stream));

    int preconditionError = 0;
    sparsePrconditionILU->tensorFactorization(m,n, nnz , d_csrRowPtrM, d_csrColIndM, d_csrValM, b, x, &preconditionError);

    if(preconditionError) {
        fprintf(stderr, "[BiCGstab/error] precondition factorization error ( singularity = %d ) \n", preconditionError);
        exit(-1);
    }


/// 0.  initial guess x
    checkCublasStatus(cublasDcopy(cublasHandle, m, x, 1, x0, 1));

/// 1. r0 = b − Ax0                              - ro = SpMV(A,x0,)   <==>  Y = a * op ( A ) * X + B * Y

/// SpMV  buffer size
    size_t reqBufferSize;
    alpha_r = -1.0;
    beta_r = 0.0;
    
    cusparseSpMVAlg_t alg = CUSPARSE_SPMV_ALG_DEFAULT;
    /// CUSPARSE_SPMV_ALG_DEFAULT = 0,
    /// CUSPARSE_SPMV_CSR_ALG1 = 2,
    /// CUSPARSE_SPMV_CSR_ALG2 = 3,
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    checkCusparseStatus(cusparseSpMV_bufferSize(handle, opA, &alpha_r, matA, vecX, &beta_r, vecY, computeType, alg, 
        &reqBufferSize));

/// SpMV resize if necessary routine buffer
    if (reqBufferSize > bufferSize) {
        /// reallocation
        if (externalBuffer) {
            checkCudaStatus(cudaFree(externalBuffer));
        }
        checkCudaStatus(cudaMallocAsync((void **)&externalBuffer, reqBufferSize, stream));
        bufferSize = reqBufferSize;
    }

/// Y = alfa * operator(A) * X + beta * Y
    beta_r = 0.0;
    alpha_r = -1.0;
    checkCusparseStatus(cusparseCreateDnVec(&vecX, m, x0, CUDA_R_64F));
    checkCusparseStatus(cusparseCreateDnVec(&vecY, m, r0, CUDA_R_64F));
    checkCusparseStatus(cusparseSpMV(handle, opA, &alpha_r, matA, vecX, &beta_r, vecY, computeType, alg, externalBuffer));
    alpha_r = 1.0;
    checkCublasStatus(cublasDaxpy(cublasHandle, m, &alpha_r, b, 1, r0, 1));

/// 2. rp = r0
    checkCublasStatus(cublasDcopy(cublasHandle, m, r0, 1, rp, 1));
    checkCublasStatus(cublasDcopy(cublasHandle, m, r0, 1, pi, 1));

/// 5. Convergence round
    int i;
    for (i = 0; i < CONVERGENCE_ROUND; ++i) {

        checkCudaStatus(cudaEventRecord(start[i], stream));
/// - 1. roi = (rp, r0)
        checkCublasStatus(cublasDdot(cublasHandle, m, rp, 1, r0, 1, &roi));
        checkCudaStatus(cudaStreamSynchronize(stream));

        /// method failed almost zero
        if(abs(roi) < 10e-24) {
            printf("method failed; roi is zero 0.0 ~ %e \n", roi);
            checkCudaStatus(cudaEventRecord(stop[i], stream));
            break;
        };

        if( i > 0) {

            if(abs(omega0) < 10e-24) {
                printf("method failed; omega0 is zero 0.0 ~ %e\n", omega0);
                checkCudaStatus(cudaEventRecord(stop[i], stream));
                break;
            }
/// - 2. beta = (roi/ro0)(a/w0)
            beta = (roi / ro0) * (alpha / omega0);

/// - 3. pi = r0 + B(p0 − omega0 * v0)			- 2x cublasDaxpy (), cublasSetZeor(pi)
            alpha_r = -omega0;
            beta_r = beta;
            checkCublasStatus(cublasDcopy(cublasHandle, m, r0, 1, pi, 1));
            checkCublasStatus(cublasDaxpy(cublasHandle, m, &alpha_r, v0, 1, p0, 1));        
            checkCublasStatus(cublasDaxpy(cublasHandle, m, &beta_r, p0, 1, pi, 1));
        }

/// - Precondtioning ~  Krylov-Space 
        /// M * ph = pi
        sparsePrconditionILU->solveSystem(m, n, nnz , d_csrRowPtrM, d_csrColIndM, d_csrValM, pi, ph, &preconditionError);

/// - 4. vi = A * ph                              - 1x SpMV CU_SPARSE
        alpha_r = 1.0;
        beta_r = 0.0;
        checkCusparseStatus(cusparseDnVecSetValues(vecX, ph));
        checkCusparseStatus(cusparseDnVecSetValues(vecY, vi));
        checkCusparseStatus(cusparseSpMV(handle, opA, &alpha_r, matA, vecX, &beta_r, vecY, computeType, alg, externalBuffer));

/// - 5. alpha =  roi / (rp, vi)                    - 1x cublasDdot()
        checkCublasStatus(cublasDdot(cublasHandle, m, rp, 1, vi, 1, &alpha));
        checkCudaStatus(cudaStreamSynchronize(stream));
        alpha = roi / alpha;

/// - 6. xi =  x0    +  alpha * ph                - 1x cublasDaxpy()               - cusparseAxpby() - graph caputre, *device/host result  [ h =  x0    +  alpha * pi ]
        alpha_r = alpha;
        checkCublasStatus(cublasDcopy(cublasHandle, m, x0, 1, xi, 1));
        checkCublasStatus(cublasDaxpy(cublasHandle, m, &alpha_r, ph, 1, xi, 1));


/// - 8. s =  r0    -  alpha * vi					- 1x cublasDaxpy()   !			s <= r0
        checkCublasStatus(cublasDcopy(cublasHandle, m, r0, 1, s, 1));
        alpha_r = -alpha;
        checkCublasStatus(cublasDaxpy(cublasHandle, m, &alpha_r, vi, 1, s, 1));


/// - 7. if is accurate enough h , set xi = h and quit  - Check convergence
    
        checkCublasStatus(cublasDnrm2(cublasHandle, m, s, 1, &nrmr));
        if(nrmr < tolerance) {
            fprintf(stdout, " Convergence ( s) : %d , nrmr %e  , relation %e\n", i, nrmr, (nrmr/nrmr0));
            checkCudaStatus(cudaEventRecord(stop[i], stream));
            break;
        }

/// - Precondtioning ~  Krylov-Space 
        /// M * sh = s
        sparsePrconditionILU->solveSystem(m, n, nnz , d_csrRowPtrM, d_csrColIndM, d_csrValM, s, sh, &preconditionError);


/// - 9. t =  A  * sh								- 1x SpMV CU_SPARSE
        alpha_r = 1.0;
        beta_r = 0.0;
        checkCusparseStatus(cusparseDnVecSetValues(vecX, sh));
        checkCusparseStatus(cusparseDnVecSetValues(vecY, t));
        checkCusparseStatus(cusparseSpMV(handle, opA, &alpha_r, matA, vecX, &beta_r, vecY, computeType, alg, externalBuffer));

/// - 10. omegai =  (t,s) / (t,t)						- 2x cublasDdot()				- cusparseSpVV() - graph capture, *device/host
        /// result
        double num = 0.0;
        double denum = 0.0;
        checkCublasStatus(cublasDdot(cublasHandle, m, t, 1, s, 1, &num));
        checkCublasStatus(cublasDdot(cublasHandle, m, t, 1, t, 1, &denum));
        checkCudaStatus(cudaStreamSynchronize(stream));
        omegai = num / denum;

/// - 11. xi =  xi  + omegai * sh						- 1x cublasDaxpy()		! -    xi <= h   ; [ xi =  h  + omegai * s ]
        alpha_r = omegai;
        //checkCublasStatus(cublasDcopy(cublasHandle, m, h, 1, xi, 1));
        checkCublasStatus(cublasDaxpy(cublasHandle, m, &alpha_r, sh, 1, xi, 1));

/// - 13. ri =  s - omegai * t
        alpha_r = -omegai;
        checkCublasStatus(cublasDcopy(cublasHandle, m, s, 1, ri, 1));
        checkCublasStatus(cublasDaxpy(cublasHandle, m, &alpha_r , t, 1, ri, 1));

/// Check convergence
        checkCublasStatus(cublasDnrm2(cublasHandle, m, ri, 1, &nrmr));
        if(nrmr < tolerance) {
            fprintf(stdout, " Convergence ( ri ): %d , nrmr %e  , relation %e\n", i, nrmr, (nrmr/nrmr0));
            checkCudaStatus(cudaEventRecord(stop[i], stream));
            break;
        }

/// - 12. if xi is accureate enough then quit
        checkCudaStatus(cudaEventRecord(stop[i], stream));

        fprintf(stdout, " iter %d , nrmr %e \n", i, nrmr);
/// 14. state transfer
        omega0 = omegai;
        ro0 = roi;
        nrmr0 = nrmr;

        double *tmp;
#define iter_swap(left, right) tmp = left; left = right; right = tmp;
        iter_swap(x0, xi) 
        iter_swap(r0, ri) 
        iter_swap(v0, vi) 
        iter_swap(p0, pi)
#undef iter_swap

    }
    validateStream;
    checkCudaStatus(cudaEventRecord(finish, stream));
    checkCudaStatus(cudaEventSynchronize(finish));

    float ms = 0.0;
    cudaEventElapsedTime(&ms, begin, finish);
    printf(" evaluation time [ms] %15.10f \n", ms);

    ms = 0.0;
    float reducedTime = 0.0;
    int j =0;
    for( ; j < i; ++j) {
        cudaEventElapsedTime(&ms, start[j], stop[j]);
        printf(" iteration ( %d ) time  %15.10f \n", j, ms);
        reducedTime += ms;
    }

    printf(" reduction time %15.10f \n", reducedTime);

/// 15. x = xi
    checkCublasStatus(cublasDcopy(cublasHandle, m, xi, 1, x, 1));
    checkCudaStatus(cudaStreamSynchronize(stream));
/// 16. free resources				- #?

        // clang-format on

        if (settings::DEBUG_CHECK_ARG) {
                checkCudaStatus(cudaStreamSynchronize(stream));
                fprintf(stdout, "[cusolverSP] nrmr = %f \n", nrmr);
        }
}

void GPUSolverSparseBiCGSTAB::allocateEvaluationBuffers(size_t m) {
        if (alloc_m < m) {
                deallocateTempBuffers();

                checkCudaStatus(cudaMalloc((void **)&r0, m * sizeof(double)));
                checkCudaStatus(cudaMalloc((void **)&ri, m * sizeof(double)));
                checkCudaStatus(cudaMalloc((void **)&rp, m * sizeof(double)));
                checkCudaStatus(cudaMalloc((void **)&x0, m * sizeof(double)));
                checkCudaStatus(cudaMalloc((void **)&xi, m * sizeof(double)));

                /// 4. v0 = p0 = 0
                checkCudaStatus(cudaMalloc((void **)&v0, m * sizeof(double)));
                checkCudaStatus(cudaMalloc((void **)&vi, m * sizeof(double)));
                checkCudaStatus(cudaMalloc((void **)&p0, m * sizeof(double)));
                checkCudaStatus(cudaMalloc((void **)&pi, m * sizeof(double)));
                checkCudaStatus(cudaMalloc((void **)&ph, m * sizeof(double))); // p^

                checkCudaStatus(cudaMalloc((void **)&s, m * sizeof(double)));
                checkCudaStatus(cudaMalloc((void **)&sh, m * sizeof(double))); // s^
                checkCudaStatus(cudaMalloc((void **)&t, m * sizeof(double)));

                alloc_m = m;
        }
}

/// <summary>
/// Deallocate temporary buffers.
/// </summary>
void GPUSolverSparseBiCGSTAB::deallocateTempBuffers() {
        if (r0) {
                /// - residual and state vector
                checkCudaStatus(cudaFree(r0));
                checkCudaStatus(cudaFree(ri));
                checkCudaStatus(cudaFree(rp));
                checkCudaStatus(cudaFree(x0));
                checkCudaStatus(cudaFree(xi));
                /// -
                checkCudaStatus(cudaFree(v0));
                checkCudaStatus(cudaFree(vi));
                checkCudaStatus(cudaFree(p0));
                checkCudaStatus(cudaFree(pi));
                checkCudaStatus(cudaFree(ph));
                /// -                
                checkCudaStatus(cudaFree(s));
                checkCudaStatus(cudaFree(sh));
                checkCudaStatus(cudaFree(t));
        }
}

GPUSolverSparseBiCGSTAB::~GPUSolverSparseBiCGSTAB() {
        cublasStatus_t cublasStatus;
        cusparseStatus_t status;
        /// ---------------------------------------------------------------------
        deallocateTempBuffers();

        if (externalBuffer) {
                checkCudaStatus(cudaFree(externalBuffer));
        }

        /// ---------------------------------------------------------------------
        checkCusparseStatus(cusparseDestroySpMat(matA));
        checkCusparseStatus(cusparseDestroyDnVec(vecX));
        checkCusparseStatus(cusparseDestroyDnVec(vecY));
        /// ---------------------------------------------------------------------
        /// cuSPARSE
        HANDLE_STATUS cusparseDestroy(handle);
        if (status) {
                fprintf(stderr, "[solver.error] destroy failure;  %s  \n", cusparseGetErrorName(status));
        }
        /// cuBLAS
        HANDLE_BLAS_STATUS cublasDestroy(cublasHandle);
        if (cublasStatus) {
                fprintf(stderr, "[cublas/error] cublas destroy failed ; %s  %s\n", cublasGetStatusName(cublasStatus), cublasGetStatusString(cublasStatus));
                exit(-1);
        }
}

void GPUSolverSparseBiCGSTAB::validateStreamState() {
        if (settings::DEBUG_CHECK_ARG) {
                /// submitted kernel into  cuda driver
                checkCudaStatus(cudaPeekAtLastError());
                /// block and wait for execution
                checkCudaStatus(cudaStreamSynchronize(stream));
        }
}

} // namespace solver

#undef HANDLE_STATUS
#undef HANDLE_BLAS_STATUS
