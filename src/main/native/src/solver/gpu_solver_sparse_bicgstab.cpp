#include "gpu_solver_sparse_bicgstab.h"

#include "../cuerror.h"
#include "../gpu_utility.h"
#include "../settings.h"

#ifdef DEBUG_GPU
#define validateStream validateStreamState()
#else
#define validateStream
#endif

namespace solver {

#define HANDLE_STATUS status =

GPUSolverSparseBiCGSTAB::GPUSolverSparseBiCGSTAB(cudaStream_t stream) : stream(stream) {
    cusolverStatus_t status;

    HANDLE_STATUS cusolverSpCreate(&handle);
    if (!status) {
        fprintf(stderr, "[solver.error] handler failure;  %s  \n", cusolverGetErrorName(status));
        exit(-1);
    }
    HANDLE_STATUS cusolverSpSetStream(handle, stream);
    if (!status) {
        fprintf(stderr, "[cusolver/error] set stream failure;  %s  \n", cusolverGetErrorName(status));
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

void GPUSolverSparseBiCGSTAB::solveSystem(int m, int n, int nnz, int *csrRowPtrA, int *csrColIndA, double *csrValA, double *b, double *x, int *singularity) {
    cusparseStatus_t status;
    if (!descrA) {
        setupTensorADescriptor();
    }

    cusolverStatus_t cusolverStatus;

    double ro[2] = {0.0, 0.0};
    double w[2] = {0.0, 0.0};
    double a = 0.0;
    double B = 0.0;

    double *xi; /// default to x0 = cublasDcopy(handle, n, xi, 1, x, 1); // nie konicznie , referencja wystarczy !
    double *r0, r1;
    double *v0, *v1, *p0, *p1;

    /// 0.  initial guess x

    /// 1. r0 = b − Ax0                              - ro = SpMV(A,x0,)   <==>  Y = a * op ( A ) * X + B * Y

    /// 2. rd = r0                                   - rd - alias ( r0 )

    /// 3. p0 = a = w0 = 1                           - scalary

    /// 4. v0 = p0 = 0                               - vector of Ns , dimension

    /// 5. For i = 1, 2, 3, …
    for (int i = 1; i < 30; i++) {

        /// ///1. ro = (rd, ri−1)                       - cublasStatus_t cublasDdot (cublasHandle_t handle, int n,
        ///                             const double          *x, int incx,
        ///                             const double          *y, int incy,
        ///                             double          *result)

        /// ///2. B = (ro[i]/ro[i−1])(a/w[i−1])                    - scalar/scalar

        /// ///3. pi = r[i−1] + B(p[i−1] − w[i−1] * v[i−1])     - 3x cublasDaxpy (), cublasSetZeor(p1)

        /// ///4. vi = Api                              - 1x SpMV CU_SPARSE

        /// ///5. a =  pi / (rd, vi)                    - 1x cublasDdot()

        /// ///6. h =  x[i-1]    +  a * pi                - 1x cublasDaxpy()

        /// ///7. if is accurate enough h , set xi = h and quit

        /// ///8. s =  r[i-1]    -  a * vi                - 1x cublasDaxpy()

        /// ///9. t =  A  * s                           - 1x SpMV CU_SPARSE

        /// ///10. wi =  (t,s) / (t,t)                  - 2x cublasDdot()

        /// ///11. xi =  h  + wi * s                    - 1x cublasDaxpy()

        /// ///12. if xi is accureate enough then quit

        /// ///13. ri =  s - wi * t
    }

    validateStream;

    if (!cusolverStatus) {
        fprintf(stderr, "[cusolver/error] csr BiCGSTAB solver operation failed ;  %s \n", cusolverGetErrorName(cusolverStatus));
        exit(-1);
    }

    if (settings::DEBUG_CHECK_ARG) {
        checkCudaStatus(cudaStreamSynchronize(stream));
        fprintf(stdout, "[cusolverSP] singularity   = %d \n", *singularity);
    }
}

void GPUSolverSparseBiCGSTAB::setupTensorADescriptor() {
    cusparseStatus_t status;
    HANDLE_STATUS cusparseCreateMatDescr(&descrA);
    if (!status) {
        fprintf(stderr, "[cusparse.error] matrix descriptor failure; %s \n", cusparseGetErrorName(status));
        exit(-1);
    }
    HANDLE_STATUS cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT);
    if (!status) {
        fprintf(stderr, "[cusparse.error] matrix descriptor set operation failure; %s \n", cusparseGetErrorName(status));
        exit(-1);
    }
    HANDLE_STATUS cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    if (!status) {
        fprintf(stderr, "[cusparse.error] matrix descriptor set operation failure; %s \n", cusparseGetErrorName(status));
        exit(-1);
    }
    HANDLE_STATUS cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    if (!status) {
        fprintf(stderr, "[cusparse.error] matrix descriptor set operation failure; %s \n", cusparseGetErrorName(status));
        exit(-1);
    }
}

GPUSolverSparseBiCGSTAB::~GPUSolverSparseBiCGSTAB() {
    cusparseStatus_t status;
    cusolverStatus_t cusolverStatus;

    if (descrA) {
        HANDLE_STATUS cusparseDestroyMatDescr(descrA);
        if (!status) {
            fprintf(stderr, "[cusparse] cusparse handle failure;  %s . %s \n", cusparseGetErrorName(status), cusparseGetErrorString(status));
        }
    }

    cusolverStatus = cusolverSpDestroy(handle);
    if (!cusolverStatus) {
        fprintf(stderr, "[cusolver/error] sovlerSP destroy;  %s  \n", cusolverGetErrorName(cusolverStatus));
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