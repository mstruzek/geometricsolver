#include "gpu_geometric_solver.h"

#include "linear_system.h"
#include "model_config.h"


namespace solver {

///=================================================================

GPUGeometricSolver::GPUGeometricSolver() : _cc(std::make_shared<GPUComputationContext>()) {}

GPUGeometricSolver::~GPUGeometricSolver() {

    if (_computation) {
        _computation.reset();
    }
    
    if (_cc) {
        _cc.reset();
    }
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

    long computationId = 0L;

    _computation = std::make_shared<GPUComputation>(computationId, _cc, std::move(points), std::move(geometrics),
                                                    std::move(constraints), std::move(parameters));
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
