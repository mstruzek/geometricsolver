#ifndef _GEOMETRIC_SOLVER_H_
#define _GEOMETRIC_SOLVER_H_

#include <vector>


void checkCudaStatus_impl(cudaError_t status, size_t __line_);

#define checkCudaStatus(status) checkCudaStatus_impl(status, __LINE__)

/**
 * @brief  setup all matricies for computation and prepare kernel stream  intertwined with cusolver
 *
 */
void solveSystemOnGPU(
    std::vector<graph::Point> const& points, 
    std::vector<graph::Geometric> const& geometrics,
    std::vector<graph::Constraint> const& constraints,
    std::vector<graph::Parameter> const& parameters, 
    std::shared_ptr<int[]> pointOffset,         /// revers mapping into point offsets
    std::shared_ptr<int[]> constraintOffset,
    std::shared_ptr<int[]> geometricOffset, 
    graph::SolverStat *stat,                    /// specific solver stats visible from JNI    
    int *err);

#endif // _GEOMETRIC_SOLVER_H_