﻿#pragma once

#include <memory>
#include <vector>

#include <atomic>
#include <condition_variable>
#include <mutex>

#include "gpu_computation_context.h"
#include "gpu_linear_system.h"
#include "model.cuh"
#include "solver_stat.h"

namespace solver {

/// Domain Model  - ( dependent on geometry state change )
class GPUComputation {

  public:
    /**
     * po zarejestrowaniu calego modelu w odpowiadajacych rejestrach , zainicjalizowac pomocnicze macierze
     *
     * przygotowac zmienne dla [cusolvera]
     *
     * przeliczenie pozycji absolutnej punktu na macierzy wyjsciowej
     *
     * commitTime --
     */
    GPUComputation(long computationId, cudaStream_t stream, std::shared_ptr<GPULinearSystem> linearSystem,
                   std::shared_ptr<GPUComputationContext> cc, std::vector<graph::Point> &&points,
                   std::vector<graph::Geometric> &&geometrics, std::vector<graph::Constraint> &&constraints,
                   std::vector<graph::Parameter> &&parameters);

    /**
     *  remove all registers containing points, constraints, parameters !
     */
    ~GPUComputation();

    ///
    /// Setup all matricies for computation and prepare kernel stream with final computation on cuSolver
    ///
    void solveSystem(solver::SolverStat *stat, cudaError_t *error);

    /**
     *
     */
    double getPointPXCoordinate(int id);

    /**
     *
     */
    double getPointPYCoordinate(int id);

    /**
     * fetch current state vector from last computation into UI
     */
    void fillPointCoordinateVector(double *stateVector);

    /**
     * update point coordinates after modifications in UI
     */
    void updatePointCoordinateVector(double stateVector[]);

    /**
     *  update constraint fixed vectors set
     */
    int updateConstraintState(int constraintId[], double vecX[], double vecY[], int size);

    /**
     *  update all parameters modified on frontend
     */
    int updateParametersValues(int parameterId[], double value[], int size);

  private:
    void preInitializeData();

    void memcpyComputationToDevice();

    void computationResultHandler(cudaStream_t stream, cudaError_t status, void *userData);

    void checkStreamNoError();

    void reportThisResult(ComputationStateData *computation);

    static void computationResultHandlerDelegate(cudaStream_t stream, cudaError_t status, void *userData);

    // std::function/ std::bind  does not provide reference to raw C pointer
    static GPUComputation *_registerComputation;

  private:
    long computationId; /// snapshotId

    std::shared_ptr<GPUComputationContext> _cc;

    std::shared_ptr<GPULinearSystem> _linearSystem;

    cudaStream_t _stream;

    cudaError_t error;

    /// mechanism for escaped data from computation tail -- first conveged computation contex or last invalid
    std::condition_variable condition;

    /// /// shared with cuda stream callback for wait, notify mechanism
    std::mutex mutex;

    /// host reference guarded by mutex
    std::atomic<ComputationStateData *> result;

  private:
    /// points register poLocations id-> point_offset
    std::vector<graph::Point> _points;

    /// geometricc register
    std::vector<graph::Geometric> _geometrics;

    /// constraints register
    std::vector<graph::Constraint> _constraints;

    /// parameters register -- paramLocation id-> param_offset
    std::vector<graph::Parameter> _parameters;

    /// Point  Offset in computation matrix [id] -> point offset   ~~ Gather Vectors
    std::vector<int> pointOffset;
    
    std::vector<int> geometricOffset;

    /// Constraint Offset in computation matrix [id] -> constraint offset
    std::vector<int> constraintOffset;

    /// Parameter Offset mapper from [id] -> parameter offset in reference dest
    std::vector<int> parameterOffset;

    /// Accymulated Geometric Object Size -- 2 * point.size()
    std::vector<int> accGeometricSize;

    /// Accumulated Constraint Size
    std::vector<int> accConstraintSize;

    size_t size;      /// wektor stanu
    size_t coffSize;  /// wspolczynniki Lagrange
    size_t dimension; /// dimension = size + coffSize

  private:
    /// device vectors
    /// Uklad rownan liniowych  [ A * x = SV ] powsta�y z linerazycji ukladu
    /// dynamicznego - tensory na urzadzeniu.
    // ( MARKER  - computation root )

    double *dev_A = nullptr;

    double *dev_b = nullptr;

    /// [ A ] * [ dx ] = [ SV ]
    double *dev_dx = nullptr;

    /// STATE VECTOR  -- lineage
    std::vector<double *> dev_SV;
    

    /// Evaluation data for  device  - CONST DATE for in process execution

    graph::Point *d_points = nullptr;
    graph::Geometric *d_geometrics = nullptr;
    graph::Constraint *d_constraints = nullptr;
    graph::Parameter *d_parameters = nullptr;

    int *d_pointOffset;
    int *d_geometricOffset;
    int *d_constraintOffset;
    int *d_parameterOffset;

    /// accumulative offset with geometric size evaluation function
    int *d_accGeometricSize;
    /// accumulative offset with constraint size evaluation function
    int *d_accConstraintSize;

    /// Solver Performance Watchers
  private:
    ///
    /// observation of submited tasks - HOST timer
    graph::StopWatchAdapter solverWatch;

    /// observation of matrix evaluation - HOST timer
    graph::StopWatchAdapter evaluationWatch;

    /// Kernel configurations - settings::get() D.3.1.1. Device-Side Kernel Launch - kernel default shared memory,

    /// number of bytes - kernel shared memory
    const unsigned int Ns = 0;


    /// grid size
    const unsigned int DIM_GRID = 1;
    const unsigned int GRID_DBG = 1;

    /// https://docs.nvidia.com/cuda/turing-tuning-guide/index.html#sm-occupancy
    /// thread block size
    const unsigned int DIM_BLOCK = 512;

    // The maximum registers per thread is 255.
    // CUDAdrv.MAX_THREADS_PER_BLOCK, which is good, ( 1024 )
    // "if your kernel uses many registers, it also limits the amount of threads you can use."
};

} // namespace solver