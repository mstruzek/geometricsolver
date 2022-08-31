#pragma once

#include "cuda_runtime_api.h"

#include "computation_state_data.h"
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

  public:
    /// cuBlas device norm2
    double *dev_norm[CMAX] = {nullptr};

    /// Local Computation References
    ComputationStateData *ev[CMAX] = {nullptr};

    /// Device Reference - `synchronized into device` one-way
    ComputationStateData *dev_ev[CMAX] = {nullptr};

  private:
    cudaStream_t stream = nullptr;

    /// === Solver Performance Watchers

    /// observation of submited tasks
    graph::StopWatchAdapter solverWatch;

    /// observation of
    graph::StopWatchAdapter evaluationWatch;

    /// observation of computation time - single computation run
    cudaEvent_t computeStart[CMAX] = {nullptr};
    cudaEvent_t computeStop[CMAX] = {nullptr};

    /// observation of matrices  preperations
    cudaEvent_t prepStart[CMAX] = {nullptr};
    cudaEvent_t prepStop[CMAX] = {nullptr};

    /// observation of accumalated cuSolver method
    cudaEvent_t solverStart[CMAX] = {nullptr};
    cudaEvent_t solverStop[CMAX] = {nullptr};
};


} // namespace solver