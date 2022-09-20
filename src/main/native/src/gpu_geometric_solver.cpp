#include "gpu_geometric_solver.h"

#include "model_config.h"

#include "cuerror.h"

namespace solver {

///=================================================================

GPUGeometricSolver::GPUGeometricSolver() {

    // main computation stream shared with CuSolver, cuBlas
    cudaStreamCreate(&stream);

    cudaError_t error = cudaPeekAtLastError();
    if (error != cudaSuccess) {
        const char *errorName = cudaGetErrorName(error);
        const char *errorStr = cudaGetErrorString(error);
        printf("[cuSolver]: stream with given error , [ %s ] %s \n", errorName, errorStr);
        exit(1);
    }

    /// Linear system solver
    solverSystem = std::make_shared<GPUSolverSystem>(stream);

    ///  Computation Context
    _computationContext = (std::make_shared<GPUComputationContext>(stream));
}

GPUGeometricSolver::~GPUGeometricSolver() {

    if (_computation) {
        _computation.reset();
    }

    solverSystem.reset();

    _computationContext.reset();
}

/**
 *
 */
void GPUGeometricSolver::registerPointType(int id, double px, double py) { points.emplace_back(id, px, py); }

/**
 *
 */
void GPUGeometricSolver::registerGeometricType(int id, int geometricTypeId, int p1, int p2, int p3, int a, int b, int c,
                                               int d) {
    geometrics.emplace_back(id, geometricTypeId, p1, p2, p3, a, b, c, d);
}

/**
 *
 */
void GPUGeometricSolver::registerParameterType(int id, double value) { parameters.emplace_back(id, value); }

/**
 *
 */
void GPUGeometricSolver::registerConstraintType(int id, int jconstraintTypeId, int k, int l, int m, int n, int paramId,
                                                double vecX, double vecY) {
    constraints.emplace_back(id, jconstraintTypeId, k, l, m, n, paramId, vecX, vecY);
}

void GPUGeometricSolver::initComputation(cudaError_t *error) {
    auto epoch = std::chrono::system_clock::now().time_since_epoch();           
    long computationId = std::chrono::duration_cast<std::chrono::milliseconds>(epoch).count();

    _computation =
        std::make_shared<GPUComputation>(computationId, stream, solverSystem, _computationContext, this);
    *error = cudaPeekAtLastError();
}

void GPUGeometricSolver::destroyComputation(cudaError_t *error) { 
    _computation.reset(); 
}

/**
 * @brief  setup all matricies for computation and prepare kernel stream  intertwined with cusolver
 *
 */
void GPUGeometricSolver::solveSystemOnGPU(SolverStat *stat, cudaError_t *error) {

    if (_computation) {
        _computation->solveSystem(stat, error);
    }
};

std::shared_ptr<GPUComputation> GPUGeometricSolver::getComputation() { return _computation; }

} // namespace solver

/// @brief Setup __device__ dest of geometric points in this moment.
///
/// @param ec evaluation context
/// @param N size of geometric object data dest
/// @param _point[] __shared__ reference into model point data
/// @tparam TZ tensor dimension without constraints
/// @return void
///
