#include "gpu_computation_context.h"

#include <algorithm>

#include "cuerror.h"
#include "utility.cuh"

namespace solver {

GPUComputationContext::GPUComputationContext(cudaStream_t stream) : stream(stream) {
    // initialize all static cuda context - no direct or indirect dependent on geometric model.

    dev_norm = std::vector<double *>(CMAX, nullptr);
    ev = std::vector<ComputationStateData *>(CMAX, nullptr);
    dev_ev = std::vector<ComputationStateData *>(CMAX, nullptr);

    computeStart = std::vector<cudaEvent_t>(CMAX, nullptr);
    computeStop = std::vector<cudaEvent_t>(CMAX, nullptr);
    prepStart = std::vector<cudaEvent_t>(CMAX, nullptr);
    prepStop = std::vector<cudaEvent_t>(CMAX, nullptr);
    solverStart = std::vector<cudaEvent_t>(CMAX, nullptr);
    solverStop = std::vector<cudaEvent_t>(CMAX, nullptr);

    for (int itr = 0; itr < CMAX; itr++) {
        // #observations
        checkCudaStatus(cudaEventCreate(&prepStart[itr]));
        checkCudaStatus(cudaEventCreate(&prepStop[itr]));
        checkCudaStatus(cudaEventCreate(&computeStart[itr]));
        checkCudaStatus(cudaEventCreate(&computeStop[itr]));
        checkCudaStatus(cudaEventCreate(&solverStart[itr]));
        checkCudaStatus(cudaEventCreate(&solverStop[itr]));
    }

    for (int itr = 0; itr < CMAX; itr++) {
        /// each computation data with its own device Evalution Context
        utility::mallocAsync(&dev_ev[itr], 1, stream);
        utility::mallocAsync(&dev_norm[itr], 1, stream);
    }

    for (int itr = 0; itr < CMAX; itr++) {
        /// each computation data with its own host Evalution Context
        utility::mallocHost(&ev[itr], 1);
    }
}

GPUComputationContext::~GPUComputationContext() {

    // !!! linear_system_method_cuSolver_reset(stream);

    for (int itr = 0; itr < CMAX; itr++) {
        /// each computation data with its own host Evalution Context
        utility::freeMemHost(&ev[itr]);
    }

    for (int itr = 0; itr < CMAX; itr++) {
        /// each computation data with its own device Evalution Context
        utility::freeAsync(dev_ev[itr], stream);
        utility::freeAsync(dev_norm[itr], stream);
    }

    for (int itr = 0; itr < CMAX; itr++) {
        // #observations
        checkCudaStatus(cudaEventDestroy(prepStart[itr]));
        checkCudaStatus(cudaEventDestroy(prepStop[itr]));
        checkCudaStatus(cudaEventDestroy(computeStart[itr]));
        checkCudaStatus(cudaEventDestroy(computeStop[itr]));
        checkCudaStatus(cudaEventDestroy(solverStart[itr]));
        checkCudaStatus(cudaEventDestroy(solverStop[itr]));
    }

    checkCudaStatus(cudaStreamSynchronize(stream));
    // implicit object for utility

    checkCudaStatus(cudaStreamDestroy(stream));
}

double *GPUComputationContext::get_dev_norm(size_t itr) { return dev_norm[itr]; }

void GPUComputationContext::info_solver_version() const {
    /// cuSolver component settings
    int major = 0;
    int minor = 0;
    int patch = 0;

    checkCuSolverStatus(cusolverGetProperty(MAJOR_VERSION, &major));
    checkCuSolverStatus(cusolverGetProperty(MINOR_VERSION, &minor));
    checkCuSolverStatus(cusolverGetProperty(PATCH_LEVEL, &patch));
    printf("[ CUSOLVER ]  version ( Major.Minor.PatchLevel): %d.%d.%d\n", major, minor, patch);
}

void GPUComputationContext::recordComputeStart(size_t itr) {
    ///
    checkCudaStatus(cudaEventRecord(computeStart[itr], stream));
}

void GPUComputationContext::recordComputeStop(size_t itr) {
    ///
    checkCudaStatus(cudaEventRecord(computeStop[itr], stream));
}

void GPUComputationContext::recordPrepStart(size_t itr) {
    ///
    checkCudaStatus(cudaEventRecord(prepStart[itr], stream));
}

void GPUComputationContext::recordPrepStop(size_t itr) {
    ///
    checkCudaStatus(cudaEventRecord(prepStop[itr], stream));
}

void GPUComputationContext::recordSolverStart(size_t itr) {
    ///
    checkCudaStatus(cudaEventRecord(solverStart[itr], stream));
}

void GPUComputationContext::recordSolverStop(size_t itr) {
    ///
    checkCudaStatus(cudaEventRecord(solverStop[itr], stream));
}

long long GPUComputationContext::getAccPrepTime(int itrBound) {
    float acc_millis = 0.0;
    float milliseconds;
    for (int itr = 0; itr <= itrBound; itr++) {
        checkCudaStatus(cudaEventSynchronize(prepStop[itr]));
        checkCudaStatus(cudaEventElapsedTime(&milliseconds, prepStart[itr], prepStop[itr]));
        acc_millis = acc_millis + milliseconds;
    }
    return (long long)(10e6 * acc_millis);
}

long long GPUComputationContext::getAccSolverTime(int itrBound) {
    float acc_millis = 0.0;
    float milliseconds;
    for (int itr = 0; itr <= itrBound; itr++) {
        checkCudaStatus(cudaEventSynchronize(solverStop[itr]));
        checkCudaStatus(cudaEventElapsedTime(&milliseconds, solverStart[itr], solverStop[itr]));
        acc_millis = acc_millis + milliseconds;
    }
    return (long long)(10e6 * acc_millis);
}

long long GPUComputationContext::getAccComputeTime(int itrBound) {
    float acc_millis = 0.0;
    float milliseconds;
    for (int itr = 0; itr <= itrBound; itr++) {
        checkCudaStatus(cudaEventSynchronize(computeStop[itr]));
        checkCudaStatus(cudaEventElapsedTime(&milliseconds, computeStart[itr], computeStop[itr]));
        acc_millis = acc_millis + milliseconds;
    }
    return (long long)(10e6 * acc_millis);
}

} // namespace solver