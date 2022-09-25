#include "gpu_solver_sparse_qr.h"

#include "../cuerror.h"
#include "../gpu_utility.h"
#include "../quda.h"
#include "../settings.h"

#ifdef DEBUG_GPU
#define validateStream validateStreamState()
#else
#define validateStream
#endif

namespace solver {

#define HANDLE_STATUS status =
GPUSolverSparseQR::GPUSolverSparseQR(cudaStream_t stream) : stream(stream) {

    cusolverStatus_t status;
    HANDLE_STATUS cusolverSpCreate(&handle);
    if (status) {
        fprintf(stderr, "[cusolver/error] handler failure; %s \n", cusolverGetErrorName(status));
        exit(-1);
    }
    HANDLE_STATUS cusolverSpSetStream(handle, stream);
    if (status) {
        fprintf(stderr, "[cusolver/error] set stream failure;  %s  \n", cusolverGetErrorName(status));
        exit(-1);
    }
}

void GPUSolverSparseQR::configure(int parameterId, int valueInt, double valueDouble) {

#define PARAMETER_SOLVER_QR_SCHEMA 1
#define PARAMETER_SOLVER_QR_TOLERANCE 2

    switch (parameterId) {
    case PARAMETER_SOLVER_QR_SCHEMA:
        // default 0 , otherwise use reduce fill-in  schema

        //#define SOLVER_QR_SCHEMA_NO 0
        //#define SOLVER_QR_SCHEMA_SYMRCM 1
        //#define SOLVER_QR_SCHEMA_SYMAMD 2
        //#define SOLVER_QR_SCHEMA_CSRMETISND 3
        schema = valueInt;
        break;
    case PARAMETER_SOLVER_QR_TOLERANCE:
        //#define SOLVER_QR_TOLERANCE 1e-10
        tolerance = valueDouble;
        break;
    default:
        return;
    }
}

void GPUSolverSparseQR::solveSystem(int m, int n, int nnz, int *csrRowPtrA, int *csrColIndA, double *csrValA, double *b, double *x, int *singularity) {
    cusparseStatus_t status;
    if (!descrA) {
        HANDLE_STATUS cusparseCreateMatDescr(&descrA);
        if (status) {
            fprintf(stderr, "[sparse.error] matrix descriptor failure;  %s . %s \n", cusparseGetErrorName(status), cusparseGetErrorString(status));
            exit(-1);
        }
        HANDLE_STATUS cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT);
        if (status) {
            fprintf(stderr, "[sparse.error] matrix descriptor set operation failure;  %s . %s \n", cusparseGetErrorName(status),
                    cusparseGetErrorString(status));
            exit(-1);
        }
        HANDLE_STATUS cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
        if (status) {
            fprintf(stderr, "[sparse.error] matrix descriptor set operation failure;  %s . %s \n", cusparseGetErrorName(status),
                    cusparseGetErrorString(status));
            exit(-1);
        }
        HANDLE_STATUS cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
        if (status) {
            fprintf(stderr, "[sparse.error] matrix descriptor set operation failure;  %s . %s \n", cusparseGetErrorName(status),
                    cusparseGetErrorString(status));
            exit(-1);
        }
    }

    cusolverStatus_t cusolverStatus;

    cusolverStatus = cusolverSpDcsrlsvqr(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tolerance, schema, x, singularity);
    validateStream;

    if (cusolverStatus != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "[cusolver/error] csr QR solver operation failed ; %s \n", cusolverGetErrorName(cusolverStatus));
        exit(-1);
    }

    if (settings::DEBUG_CHECK_ARG) {
        checkCudaStatus(cudaStreamSynchronize(stream));
        fprintf(stdout, "[cusolverSP] singularity   = %d \n", *singularity);
    }
}

GPUSolverSparseQR::~GPUSolverSparseQR() {
    cusparseStatus_t status;

    if (descrA) {
        HANDLE_STATUS cusparseDestroyMatDescr(descrA);
        if (status) {
            fprintf(stderr, "[cusparse] cusparse handle failure; %s . %s \n", cusparseGetErrorName(status), cusparseGetErrorString(status));
        }
    }
    cusolverStatus_t cusolverStatus;
    cusolverStatus = cusolverSpDestroy(handle);
    if (cusolverStatus) {
        fprintf(stderr, "[cusolver/error] sovlerSP destroy; ( %s ) \n", cusolverGetErrorName(cusolverStatus));
    }
}

void GPUSolverSparseQR::validateStreamState() {
    if (settings::DEBUG_CHECK_ARG) {
        /// submitted kernel into  cuda driver
        checkCudaStatus(cudaPeekAtLastError());
        /// block and wait for execution
        checkCudaStatus(cudaStreamSynchronize(stream));
    }
}

#undef HANDLE_STATUS

} // namespace solver