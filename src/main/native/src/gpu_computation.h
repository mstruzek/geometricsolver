#ifndef _GPU_COMPUTATION_H_
#define _GPU_COMPUTATION_H_

#include <memory>
#include <vector>

#include <atomic>
#include <condition_variable>
#include <mutex>

#include <cusparse.h>

#include "gpu_computation_context.h"
#include "gpu_solver_system.h"
#include "kernel_traits.h"

#include "model.cuh"
#include "solver_stat.h"

#include "tensor_operation.h"
#include "format_encoder.h"

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
    GPUComputation(long computationId, cudaStream_t stream, std::shared_ptr<GPUSolverSystem> solverSystem,
                   std::shared_ptr<GPUComputationContext> cc, GPUGeometricSolver *solver);

    /**
     *  remove all registers containing points, constraints, parameters !
     */
    ~GPUComputation();

    /// <summary>
    /// Each iteration has different matrix A computation characteristics.
    /// </summary>
    /// <param name="round"></param>
    /// <returns></returns>
    ComputationLayout selectComputationLayout(int round);

    /// <summary>
    /// Post process tensor A computation into CSR format from sparse COO format .
    /// Applied to sparse tensor only.
    /// </summary>
    /// <param name="round"></param>
    /// <param name="computationLayout"></param>
    void PostProcessTensorA(int round, ComputationLayout computationLayout);

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

    void SetComputationHandler(ComputationState *compState);

    void validateStreamState();

    void PreInitializeComputationState(ComputationState *ev, int itr, ComputationLayout computationLayout);

    void InitializeStateVector();

    int getSharedMemoryRequirments(ComputationLayout computationLayout);

    void DeviceConstructTensorA(ComputationState *dev_ev, ComputationLayout computationLayout, cudaStream_t stream);

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

    SolverMode solverMode;

    std::shared_ptr<GPUComputationContext> computationContext;

    std::shared_ptr<solver::GPUSolverSystem> solverSystem;
   
    /// conversion from COO to CSR format for linear sparse solver
    TensorOperation tensorOperation;

    /// COO journal format conversion to CSR canonical form.
    FormatEncoder formatEncoder;


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
    utility::cu_vector<graph::Point> points;

    /// geometricc register
    utility::cu_vector<graph::Geometric> geometrics;

    /// constraints register
    utility::cu_vector<graph::Constraint> constraints;

    /// parameters register -- paramLocation id-> param_offset
    utility::cu_vector<graph::Parameter> parameters;

    /// Point  Offset in computation matrix [id] -> point offset   ~~ Gather Vectors
    utility::cu_vector<int> pointOffset;

    utility::cu_vector<int> geometricOffset;

    /// Constraint Offset in computation matrix [id] -> constraint offset
    utility::cu_vector<int> constraintOffset;

    /// Parameter Offset mapper from [id] -> parameter offset in reference dest
    utility::cu_vector<int> parameterOffset;

    /// Accymulated Geometric Object Size -- 2 * point.size()
    utility::cu_vector<int> accGeometricSize;

    /// Accumulated Constraint Size
    utility::cu_vector<int> accConstraintSize;

    /// Accumulated Writes in COO format from kernel into Stiff Tensor
    utility::cu_vector<int> accCooWriteStiffTensor;

    /// Accumulated Writes in COO format from kernel into Jacobian Tensor
    utility::cu_vector<int> accCooWriteJacobianTensor;

    /// Accumulated Writes in COO format from kernel into Hessian Tensor
    utility::cu_vector<int> accCooWriteHessianTensor;

    int size;      /// wektor stanu
    int coffSize;  /// wspolczynniki Lagrange
    int dimension; /// dimension = size + coffSize

  private:
    /// device vectors
    /// Uklad rownan liniowych  [ A * x = SV ] powsta�y z linerazycji ukladu
    /// dynamicznego - tensory na urzadzeniu.
    // ( MARKER  - computation root )

    utility::dev_vector<double> dev_A;

    utility::dev_vector<double> dev_b;

    /// [ A ] * [ dx ] = [ SV ]
    utility::dev_vector<double> dev_dx;

    /// STATE VECTOR  -- lineage
    std::vector<utility::dev_vector<double>> dev_SV;

    /// Evaluation data for  device  - CONST DATE for in process execution

    utility::dev_vector<graph::Point> d_points;
    utility::dev_vector<graph::Geometric> d_geometrics;
    utility::dev_vector<graph::Constraint> d_constraints;
    utility::dev_vector<graph::Parameter> d_parameters;

    utility::dev_vector<int> d_pointOffset;
    utility::dev_vector<int> d_geometricOffset;
    utility::dev_vector<int> d_constraintOffset;
    utility::dev_vector<int> d_parameterOffset;

    /// accumulative offset with geometric size evaluation function
    utility::dev_vector<int> d_accGeometricSize;
    /// accumulative offset with constraint size evaluation function
    utility::dev_vector<int> d_accConstraintSize;

    /// ( ralative offset ) Accumulated Writes in COO format from kernel into Stiff Tensor
    utility::dev_vector<int> d_accCooWriteStiffTensor;

    /// ( ralative offset ) Accumulated Writes in COO format from kernel into Jacobian Tensor
    utility::dev_vector<int> d_accCooWriteJacobianTensor;

    /// ( ralative offset ) Accumulated Writes in COO format from kernel into Jacobian Tensor
    utility::dev_vector<int> d_accCooWriteHessianTensor;

    /// non-zero elements in coo/scr tensor A
    int nnz;

    /// non-zero elements in csr format for sparse solver ( this is for compuation ). 
    int nnz_sv;

    /// offset value for kernel Jacobian writes
    int cooWritesStiffSize;

    /// offset for kernel Transposed Jacobian writes
    int cooWirtesJacobianSize;

    /// not-transformed row vector of indicies, Coordinate Format COO ( initialized once )
    utility::dev_vector<int> d_cooRowInd;

    /// not-transformed column vector of indicies, Coordinate Format COO ( initialized once )
    utility::dev_vector<int> d_cooColInd;

    /// transformed in first-iteration - in ordered [d_cooRowInd_order, d_csrColIndA, d_cooVal]
    utility::dev_vector<int> d_cooRowInd_order;

    /// transformed in first-iteration -->  directly to csrColInd
    // int *d_cooColInd_tmp = NULL;

    /// COO vector of values, Coordinate Format COO, or CSR format sorted
    utility::dev_vector<double> d_cooVal;

    /// CSR tensor A rows (compressed), Compressed Sparsed Row Format CSR ;  Xcoosort  | cooTcsr
    utility::dev_vector<int> d_csrRowPtrA;

    /// CSR tensor A columns, Compressed Sparsed Row Format CSR ; Xcoosort
    utility::dev_vector<int> d_csrColIndA;

    /// CSR tensor A values, Compressed Sparsed Row Format CSR ; perm(d_cooValA)
    utility::dev_vector<double> d_csrValA;

    /// Permutation vector "i" - store into , gather from  P[i]
    utility::dev_vector<int> d_PT;

    /// inversed permutation vector INVP[i] - store into, gather from "i"
    utility::dev_vector<int> d_INV_PT;

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