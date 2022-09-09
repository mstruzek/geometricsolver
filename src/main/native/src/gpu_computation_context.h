#ifndef _GPU_COMPUTATION_CONTEXT_H_
#define _GPU_COMPUTATION_CONTEXT_H_

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

    void ComputeStart(size_t itr);

    void ComputeStop(size_t itr);

    void PrepStart(size_t itr);

    void PrepStop(size_t itr);

    void SolverStart(size_t itr);

    void SolverStop(size_t itr);

    void info_solver_version() const;
        
    long long getAccPrepTime(int itrBound); 

    long long getAccSolverTime(int itrBound); 

    long long getAccComputeTime(int itrBound); 

    ComputationState* host_ev(int iteration);

    ComputationState* get_dev_ev(int iteration);

  public:
    /// cuBlas device norm2
    std::vector<double *> dev_norm;

  private:
    cudaStream_t stream = nullptr;

    /// Local Computation References
    std::vector<ComputationState *> ev;

    /// Device Reference - `synchronized into device` one-way
    std::vector<ComputationState *> dev_ev;

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

#endif // _GPU_COMPUTATION_CONTEXT_H_