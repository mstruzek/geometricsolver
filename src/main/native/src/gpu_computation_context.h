#pragma once

#include "cuda_runtime_api.h"

#include <vector>
#include "computation_state.cuh"
#include "stop_watch.h"

/// MAX SOLVER ITERATIONS
#define CMAX 20

namespace solver {

/*
 * == = Solver Structural Data - (no dependency on model geometry or constraints)
 */
class GPUComputationContext {

  public:
    GPUComputationContext(cudaStream_t stream);

    /**
     *  workspace - zwolnic pamiec i wyzerowac wskazniki , \\cusolver
     */
    ~GPUComputationContext();

    double *get_dev_norm(size_t itr);

    void recordComputeStart(size_t itr);

    void recordComputeStop(size_t itr);

    void recordPrepStart(size_t itr);

    void recordPrepStop(size_t itr);

    void recordSolverStart(size_t itr);

    void recordSolverStop(size_t itr);

    void info_solver_version() const;
        
    long long getAccPrepTime(int itrBound); 

    long long getAccSolverTime(int itrBound); 

    long long getAccComputeTime(int itrBound); 

  public:
    /// cuBlas device norm2
    std::vector<double *> dev_norm;

    /// Local Computation References
    std::vector<ComputationState *> ev;

    /// Device Reference - `synchronized into device` one-way
    std::vector<ComputationState *> dev_ev;

  private:
    cudaStream_t stream = nullptr;

    /// === Solver Performance Watchers

    /// observation of submited tasks
    graph::StopWatchAdapter solverWatch;

    /// observation of
    graph::StopWatchAdapter evaluationWatch;

    /// observation of computation time - single computation run
    std::vector<cudaEvent_t> computeStart;
    std::vector<cudaEvent_t> computeStop;

    /// observation of matrices  preperations
    std::vector<cudaEvent_t> prepStart;
    std::vector<cudaEvent_t> prepStop;

    /// observation of accumalated cuSolver method
    std::vector<cudaEvent_t> solverStart;
    std::vector<cudaEvent_t> solverStop;
};

} // namespace solver