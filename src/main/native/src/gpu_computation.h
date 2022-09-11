#ifndef _GPU_COMPUTATION_H_
#define _GPU_COMPUTATION_H_

#include <memory>
#include <vector>

#include <atomic>
#include <condition_variable>
#include <mutex>

#include <cusparse.h>

#include "gpu_computation_context.h"
#include "gpu_linear_system.h"
#include "kernel_traits.h"

#include "model.cuh"
#include "solver_stat.h"

#include "tensor_operation.h"

#include "gpu_geometric_solver.h"


#define BLOCK_DIM 512

#define OBJECTS_PER_THREAD 1

#define ELEMENTS_PER_THREAD 4

namespace solver {

class GPUGeometricSolver;

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
                   std::shared_ptr<GPUComputationContext> cc, GPUGeometricSolver *solver);

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

    void AddComputationHandler(ComputationState *compState);

    void validateStreamState();

    void PreInitializeComputationState(ComputationState *ev, int itr);

    void InitializeStateVector();

    void DeviceConstructTensorA(ComputationState *dev_ev, cudaStream_t stream);

    void DeviceConstructTensorB(ComputationState *dev_ev, cudaStream_t stream);

    void DebugTensorConstruction(ComputationState *dev_ev);

    void DebugTensorSV(ComputationState *dev_ev);

    void StateVectorAddDelta(double *dev_SV, double *dev_b);

    void RebuildPointsFrom(ComputationState *computation);

    void CudaGraphCaptureAndLaunch(cudaGraphExec_t graphExec);

    void BuildSolverStat(ComputationState *computation, SolverStat *stat);

    void ReportComputationResult(ComputationState *computation);

    static void computationResultHandlerDelegate(cudaStream_t stream, cudaError_t status, void *userData);

    static void registerComputation(GPUComputation *computation);

    static void unregisterComputation(GPUComputation *computation);

    // std::function/ std::bind  does not provide reference to raw C pointer
    static GPUComputation *_registerComputation;

    /// number of bytes - kernel shared memory
    const unsigned int Ns = 0;

    /// grid size
    constexpr static unsigned int DIM_GRID = 1;
    constexpr static unsigned int GRID_DBG = 1;

    /// https://docs.nvidia.com/cuda/turing-tuning-guide/index.html#sm-occupancy
    /// thread block size
    const unsigned int DIM_BLOCK = 512;
    
    // CUDAdrv.MAX_THREADS_PER_BLOCK, which is good, ( 1024 )
    // "if your kernel uses many registers, it also limits the amount of threads you can use."

  private:
    const long computationId; /// snapshotId

    ComputationMode computationMode;


    std::shared_ptr<GPUComputationContext> computationContext;

    std::shared_ptr<GPULinearSystem> linearSystem;


    /// conversion from COO to CSR format for linear sparse solver
    TensorOperation tensorOperation;

    cudaStream_t stream;

    cudaError_t error;

    /// mechanism for escaped data from computation tail -- first conveged computation contex or last invalid
    std::condition_variable condition;

    /// /// shared with cuda stream callback for wait, notify mechanism
    std::mutex mutex;

    /// host reference guarded by mutex
    std::atomic<ComputationState *> result;

  private:
    /// points register poLocations id-> point_offset
    std::vector<graph::Point> points;

    /// geometricc register
    std::vector<graph::Geometric> geometrics;

    /// constraints register
    std::vector<graph::Constraint> constraints;

    /// parameters register -- paramLocation id-> param_offset
    std::vector<graph::Parameter> parameters;

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

    /// Accumulated Writes in COO format from kernel into Stiff Tensor
    std::vector<int> accCooWriteStiffTensor;

    /// Accumulated Writes in COO format from kernel into Jacobian Tensor
    std::vector<int> accCooWriteJacobianTensor;

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

    graph::Point* d_points = NULL;
    graph::Geometric *d_geometrics = NULL;
    graph::Constraint *d_constraints = NULL;
    graph::Parameter *d_parameters = NULL;

    int *d_pointOffset;
    int *d_geometricOffset;
    int *d_constraintOffset;
    int *d_parameterOffset;

    /// accumulative offset with geometric size evaluation function
    int *d_accGeometricSize;
    /// accumulative offset with constraint size evaluation function
    int *d_accConstraintSize;

    /// ( ralative offset ) Accumulated Writes in COO format from kernel into Stiff Tensor
    int *d_accCooWriteStiffTensor;

    /// ( ralative offset ) Accumulated Writes in COO format from kernel into Jacobian Tensor
    int *d_accCooWriteJacobianTensor;

    /// ( ralative offset ) Accumulated Writes in COO format from kernel into Jacobian Tensor
    int *d_accCooWriteHessianTensor;

    /// non-zero elements in coo/scr tensor A
    int nnz;

    /// offset value for kernel Jacobian writes
    int cooWritesStiffSize;

    /// offset for kernel Transposed Jacobian writes
    int cooWirtesJacobianSize;

    /// not-transformed row vector of indicies, Coordinate Format COO ( initialized once )
    int *d_cooRowInd = NULL;

    /// not-transformed column vector of indicies, Coordinate Format COO ( initialized once )
    int *d_cooColInd = NULL;

    /// transformed in first-iteration
    int *d_cooRowInd_tmp = NULL;

    /// transformed in first-iteration -->  directly to csrColInd
    // int *d_cooColInd_tmp = NULL;



    /// COO vector of values, Coordinate Format COO, or CSR format sorted
    double *d_cooVal = NULL;

    /// CSR tensor A rows (compressed), Compressed Sparsed Row Format CSR ;  Xcoosort  | cooTcsr 
    int *d_csrRowPtrA;

    /// CSR tensor A columns, Compressed Sparsed Row Format CSR ; Xcoosort
    int *d_csrColIndA;

    /// CSR tensor A values, Compressed Sparsed Row Format CSR ; perm(d_cooValA)
    double *d_csrValA;

    /// Permutation vector "i" - store into , gather from  P[i]
    int *d_PT = NULL;

    /// inversed permutation vector INVP[i] - store into, gather from "i"
    int *d_INV_PT = NULL;


    /// Solver Performance Watchers
  private:
    ///
    /// observation of submited tasks - HOST timer
    graph::StopWatchAdapter solverWatch;

    /// observation of matrix evaluation - HOST timer
    graph::StopWatchAdapter evaluationWatch;

    /// Kernel configurations - settings::get() D.3.1.1. Device-Side Kernel Launch - kernel default shared memory,

};

#undef BLOCK_DIM

#undef OBJECTS_PER_THREAD

#undef ELEMENTS_PER_THREAD


} // namespace solver

#endif // _GPU_COMPUTATION_H_