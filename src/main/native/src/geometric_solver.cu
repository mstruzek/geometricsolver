#include "geometric_solver.h"

#include "cuda_runtime.h"
#include "stop_watch.h"

#include <atomic>
#include <condition_variable>
#include <memory>

#include <algorithm>
#include <functional>
#include <numeric>

#include <stdexcept>
#include <stdio.h>

#include "model.cuh"
#include "solver_kernel.cuh"

#include "linear_system.h"

#include "model_config.h"

#include "settings.h"

/// MAX SOLVER ITERATIONS
#define CMAX 20

#define CONVERGENCE_LIMIT 10e-5

/// ================== Domain Model  - ( dependent on geometry state change )

// -- przeniescie do structury/classy GpuGeometricSolver {} ~dtor ctor

/// points register
static std::vector<graph::Point> points; /// poLocations id-> point_offset

/// geometricc register
static std::vector<graph::Geometric> geometrics; /// ==> Macierz A, accumulative offset for each primitive

/// constraints register
static std::vector<graph::Constraint> constraints; /// ===> Wiezy , accumulative offset for each constraint

/// parameters register
static std::vector<graph::Parameter> parameters; /// paramLocation id-> param_offset

/// Point  Offset in computation matrix [id] -> point offset   ~~ Gather Vectors
static std::vector<int> pointOffset;

/// Constraint Offset in computation matrix [id] -> constraint offset 
static std::vector<int> constraintOffset;

/// Parameter Offset mapper from [id] -> parameter offset in reference dest
static std::vector<int> parameterOffset;

/// Accymulated Geometric Object Size
static std::vector<int> accGeometricSize; /// 2 * point.size()

/// Accumulated Constraint Size
static std::vector<int> accConstraintSize;

static size_t size;      /// wektor stanu
static size_t coffSize;  /// wspolczynniki Lagrange
static size_t dimension; /// dimension = size + coffSize

/// Uklad rownan liniowych  [ A * x = SV ] powsta�y z linerazycji ukladu
/// dynamicznego - tensory na urzadzeniu.
// ( MARKER  - computation root )

#define COMPUTATION_ROOT

double *dev_A = nullptr;

double *dev_b = nullptr;

/// [ A ] * [ dx ] = [ SV ]
double *dev_dx = nullptr;

/// STATE VECTOR  -- lineage
double *dev_SV[CMAX] = {NULL};

/// Evaluation data for  device  - CONST DATE for in process execution

graph::Point *d_points = nullptr;
graph::Geometric *d_geometrics = nullptr;
graph::Constraint *d_constraints = nullptr;
graph::Parameter *d_parameters = nullptr;

int *d_pointOffset;

int *d_constraintOffset;

int *d_parameterOffset;

/// accumulative offset with geometric size evaluation function
int *d_accGeometricSize;

/// accumulative offset with constraint size evaluation function
int *d_accConstraintSize;

/// =================== Solver Structural Data - ( no dependency on model geometry or constraints )

/// ===========================================================
/// Async Computation - Implicit ref in utility:: namespace
///
static cudaStream_t stream = nullptr;

/// cuBlas device norm2
static double *dev_norm[CMAX] = {NULL};

/// Local Computation References
static ComputationState *ev[CMAX] = {NULL};

/// Device Reference - `synchronized into device` one-way
static ComputationState *dev_ev[CMAX] = {NULL};

/// === Solver Performance Watchers

/// observation of submited tasks
static graph::StopWatchAdapter solverWatch;

/// observation of 
static graph::StopWatchAdapter evaluationWatch;

/// observation of computation time - single computation run
static cudaEvent_t computeStart[CMAX] = {nullptr};
static cudaEvent_t computeStop[CMAX] = {nullptr};

/// observation of matrices  preperations
static cudaEvent_t prepStart[CMAX] = {nullptr};
static cudaEvent_t prepStop[CMAX] = {nullptr};

/// observation of accumalated cuSolver method
static cudaEvent_t solverStart[CMAX] = {nullptr};
static cudaEvent_t solverStop[CMAX] = {nullptr};

/// ==============================================================================

/// Kernel configurations - settings::get() D.3.1.1. Device-Side Kernel Launch - kernel default shared memory, number of
/// bytes
constexpr size_t Ns = 0;

/// grid size
constexpr size_t DIM_GRID = 1;
constexpr size_t GRID_DBG = 1;

/*
    The maximum registers per thread is 255.

    CUDAdrv.MAX_THREADS_PER_BLOCK, which is good, ( 1024 )

    "if your kernel uses many registers, it also limits the amount of threads
   you can use."

*/

/// https://docs.nvidia.com/cuda/turing-tuning-guide/index.html#sm-occupancy
///
/// thread block size
constexpr size_t DIM_BLOCK = 512;

/// mechanism for escaped data from computation tail -- first conveged computation contex or last invalid

/// shared with cuda stream callback for wait, notify mechanism
std::condition_variable condition;

///
std::mutex mutex;

/// host reference guarded by mutex
std::atomic<ComputationState *> result;

#define CDEBUG

/// implicitly depends on `stream` reference !
#include "utility.cuh"

#define CDEBUG
#undef CDEBUG

///
/// ===========================================================

namespace solver {

void initComputationContext(cudaError_t *error) {

    // initialize all static cuda context - no direct or indirect dependent on geometric model.
    //

    /// cuSolver component settings
    int major = 0;
    int minor = 0;
    int patch = 0;

    checkCuSolverStatus(cusolverGetProperty(MAJOR_VERSION, &major));
    checkCuSolverStatus(cusolverGetProperty(MINOR_VERSION, &minor));
    checkCuSolverStatus(cusolverGetProperty(PATCH_LEVEL, &patch));
    printf("[ CUSOLVER ]  version ( Major.Minor.PatchLevel): %d.%d.%d\n", major, minor, patch);

    if (stream == nullptr) {
        // implicit in utility::
        checkCudaStatus(cudaStreamCreate(&stream));

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
            utility::mallocAsync(&dev_ev[itr], 1);
            utility::mallocAsync(&dev_norm[itr], 1);
        }

        for (int itr = 0; itr < CMAX; itr++) {
            /// each computation data with its own host Evalution Context
            utility::mallocHost(&ev[itr], 1);
        }
    }
}

/**
 *
 */
void destroyComputationContext(cudaError_t *error) {

    if (stream != nullptr) {

        linear_system_method_cuSolver_reset(stream);

        for (int itr = 0; itr < CMAX; itr++) {
            /// each computation data with its own host Evalution Context
            utility::freeMemHost(&ev[itr]);                       
        }

        for (int itr = 0; itr < CMAX; itr++) {
            /// each computation data with its own device Evalution Context
            utility::freeMem(&dev_ev[itr]);
            utility::freeMem(&dev_norm[itr]);
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
        stream = nullptr;

        *error = cudaSuccess;
    }
}

/**
 *  Last commit time accessor !!!
 */
long getCommitTime() { return 0L; }

/**
 *
 */
void initComputation(cudaError_t *error) {
    if (points.size() == 0) {
        throw new std::exception("empty solution space, add some geometric types");
    }

    // model aggreate for const immutable data
    if (points.size() == 0) {
        printf("[solver] - empty evaluation space model\n");
        *error = cudaSuccess;
        return;
    }

    if (constraints.size() == 0) {
        printf("[solver] - no constraint configuration applied onto model\n");
        *error = cudaSuccess;
        return;
    }

    evaluationWatch.setStartTick();

    if (dev_A != nullptr)
        return;      


    /// mapping from point Id => point dest offset
    pointOffset = utility::stateOffset(points, [](auto point) { return point->id; });
    constraintOffset = utility::stateOffset(constraints, [](auto constraint) { return constraint->id; });
    /// this is mapping from Id => parameter dest offset
    parameterOffset = utility::stateOffset(parameters, [](auto parameter) { return parameter->id; });
    /// accumalted position of geometric block
    accGeometricSize = utility::accumalatedValue(geometrics, graph::geometricSetSize);
    /// accumulated position of constrain block
    accConstraintSize = utility::accumalatedValue(constraints, graph::constraintSize);

    /// `A` tensor internal structure dimensions
    size = std::accumulate(geometrics.begin(), geometrics.end(), 0,
                           [](auto acc, auto const &geometric) { return acc + graph::geometricSetSize(geometric); });

    coffSize = std::accumulate(constraints.begin(), constraints.end(), 0, [](auto acc, auto const &constraint) {
        return acc + graph::constraintSize(constraint);
    });

    dimension = size + coffSize;

    /// ============================================================
    ///         Host Computation with references to Device
    /// ============================================================

    /// const data in computation
    utility::mallocAsync(&d_points, points.size());
    utility::mallocAsync(&d_geometrics, geometrics.size());
    utility::mallocAsync(&d_constraints, constraints.size());
    utility::mallocAsync(&d_parameters, parameters.size());

    utility::mallocAsync(&d_pointOffset, pointOffset.size());
    utility::mallocAsync(&d_constraintOffset, constraintOffset.size());
    utility::mallocAsync(&d_parameterOffset, parameterOffset.size());
    utility::mallocAsync(&d_accGeometricSize, geometrics.size());
    utility::mallocAsync(&d_accConstraintSize, constraints.size());

    // immutables -
    utility::memcpyToDevice(&d_points, points);
    utility::memcpyToDevice(&d_geometrics, geometrics);
    utility::memcpyToDevice(&d_constraints, constraints);
    utility::memcpyToDevice(&d_parameters, parameters);

    // immutables
    utility::memcpyToDevice(&d_pointOffset, pointOffset.data(), pointOffset.size());
    utility::memcpyToDevice(&d_constraintOffset, constraintOffset.data(), constraintOffset.size());

    utility::memcpyToDevice(&d_accGeometricSize, accGeometricSize.data(), accGeometricSize.size());
    utility::memcpyToDevice(&d_accConstraintSize, accConstraintSize.data(), accConstraintSize.size());
    if (!parameters.empty()) {
        utility::memcpyToDevice(&d_parameterOffset, parameterOffset.data(), parameterOffset.size());
    }

    size_t N = dimension;
    ///
    ///  [ GPU ] tensors `A` `x` `dx` `b`  ------------------------
    ///
    utility::mallocAsync(&dev_A, N * N);
    utility::mallocAsync(&dev_b, N);
    utility::mallocAsync(&dev_dx, N);

    for (int itr = 0; itr < CMAX; itr++) {
        /// each computation data with its own StateVector
        utility::mallocAsync(&dev_SV[itr], N);
    }

    printf("[solver] model stage consistent !\n");

    *error = cudaSuccess;
}

void destroyComputation(cudaError_t *error) {

    /// z javy  z rejestracji
    points.clear();
    geometrics.clear();
    constraints.clear();
    parameters.clear();

    /// at least one solver computation
    if (dev_A == nullptr) {
        *error = cudaSuccess;
        return;
    }

    for (int itr = 0; itr < CMAX; itr++) {
        utility::freeMem(&dev_SV[itr]);
    }

    utility::freeMem(&dev_A);
    utility::freeMem(&dev_b);
    utility::freeMem(&dev_dx);

    utility::freeMem(&d_parameterOffset);
    utility::freeMem(&d_accGeometricSize);
    utility::freeMem(&d_accConstraintSize);
    utility::freeMem(&d_constraintOffset);
    utility::freeMem(&d_pointOffset);

    utility::freeMem(&d_points);
    utility::freeMem(&d_geometrics);
    utility::freeMem(&d_constraints);
    utility::freeMem(&d_parameters);

    pointOffset.clear();
    constraintOffset.clear();
    parameterOffset.clear();
    accGeometricSize.clear();
    accConstraintSize.clear();

    dimension = 0;
    size = 0;
    coffSize = 0;

    printf("[solver] model stage consistent !\n");
}

/**
 *
 */
void registerPointType(int id, double px, double py) {
    points.emplace_back(id, px, py);
}

/**
 *
 */
void registerGeometricType(int id, int geometricTypeId, int p1, int p2, int p3, int a, int b, int c, int d) {
    geometrics.emplace_back(id, geometricTypeId, p1, p2, p3, a, b, c, d);
}

/**
 *
 */
void registerParameterType(int id, double value) {
    parameters.emplace_back(id, value);
}

/**
 *
 */
void registerConstraintType(int id, int jconstraintTypeId, int k, int l, int m, int n, int paramId, double vecX,
                            double vecY) {
    constraints.emplace_back(id, jconstraintTypeId, k, l, m, n, paramId, vecX, vecY);
}

/**
 *
 */
double getPointPXCoordinate(int id) {
    int offset = pointOffset[id];
    double px = points[offset].x;
    return px;
}

/**
 *
 */
double getPointPYCoordinate(int id) {
    int offset = pointOffset[id];
    double py = points[offset].y;
    return py;
}

void fillPointCoordinateVector(double *stateVector) {
    for (size_t i = 0, size = points.size(); i < size; i++) {
        graph::Point &p = points[i];
        stateVector[2 * i] = p.x;
        stateVector[2 * i + 1] = p.y;
    }
}


int updateConstraintState(int *constraintId, double *vecX, double *vecY, int size) {

    evaluationWatch.setStartTick();

    for (int itr = 0; itr < size; itr++) {
        int cId = constraintId[itr];
        int offset  = constraintOffset[cId];               
        graph::Constraint &constraint = constraints[offset];

        if (constraint.constraintTypeId != CONSTRAINT_TYPE_ID_FIX_POINT) {
            printf("[error] constraint type only supported is ConstraintFixPoint ! \n");
            return 1;
        }
        constraint.vecX = vecX[itr];
        constraint.vecY = vecY[itr];
    }

    if (d_constraints != nullptr) {
        utility::memcpyToDevice(&d_constraints, constraints);
    }
    return 0;   
}

int updateParametersValues(int *parameterId, double *value, int size) {
    for (int itr = 0; itr < size; itr++) {
        int pId = parameterId[itr];
        int offset = parameterOffset[pId];
        graph::Parameter &parameter = parameters[offset];
        parameter.value = value[itr];        
    }

    if (d_parameters != nullptr) {
        utility::memcpyToDevice(&d_parameters, parameters);
    }
    return 0;   
}

void updatePointCoordinateVector(double *stateVector) {
    for (size_t i = 0, size = points.size(); i < size; i++) {
        graph::Point &p = points[i];
        p.x = stateVector[2 * i];
        p.y = stateVector[2 * i + 1];
    }

    if (d_points != nullptr) {
        utility::memcpyToDevice(&d_points, points);        
    }
}

/// <summary>
///
/// </summary>
/// <param name="ev">[__host__]</param>
/// <param name="offset">[__device__]</param>
/// <param name="stream"></param>
void ConstraintGetFullNorm(size_t coffSize, size_t size, double *b, double *result, cudaStream_t stream) {
    linear_system_method_cuBlas_vectorNorm(static_cast<int>(coffSize), (b + size), result, stream);
}


/// <summary>
/// Computation Round Handler
/// </summary>
/// <param name="userData"></param>
void computationResultHandler(cudaStream_t stream, cudaError_t status, void *userData) {
    // synchronize on stream
    ComputationState *computation = static_cast<ComputationState *>(userData);

    // obsluga bledow w strumieniu
    if (status != cudaSuccess) {
        const char *errorName = cudaGetErrorName(status);
        const char *errorStr = cudaGetErrorString(status);

        printf("[error] - computation id [%d] ,  %s = %s \n", computation->cID, errorName, errorStr);
        return;
    }

    if (settings::get()->DEBUG) {
        printf("[ resutl / handler ]- computationId (%d)  , norm (%e) \n", computation->cID, computation->norm);
    }

    bool last = computation->cID == (CMAX - 1);

    const double CONVERGENCE = settings::get()->CU_SOLVER_EPSILON;

    if (computation->norm < CONVERGENCE  || last) {
        /// update offset        

        auto desidreState = static_cast<ComputationState *>(nullptr);
        if (result.compare_exchange_strong(desidreState, computation)) {
            condition.notify_one();
        }
    }

    // synchronize with stream next computation
}

#undef CDEBUG

///
/// Setup all matricies for computation and prepare kernel stream  intertwined
/// with cuSolver
///
///
void solveSystemOnGPU(solver::SolverStat *stat, cudaError_t *error) {
    /// # Consideration -> ::"ingest stream and observe" fetch first converged
    ///
    ///
    ///   -- "cublas_v2.h" funkcje API czytaj/i wpisuja  wspolczynnik z HOST jak i
    ///   z DEVICE
    ///
    ///   -- zalaczamy zadanie FI(q) = 0 norm -> wpiszemy do ExecutionContext
    ///   variable
    ///
    ///   --- data-dest lineage memcpy(deviceToDevice)
    ///

    //- fill in A , b, SV

    size_t N = dimension;

    /// !!!  max(max(points.size(), geometrics.size()), constraints.size());

    /// default kernel settings
    const size_t ST_DIM_GRID = settings::get()->GRID_SIZE;
    const size_t ST_DIM_BLOCK = settings::get()->BLOCK_SIZE;

    solverWatch.setStartTick();

    /// prepare local offset context
    result.store(NULL, std::memory_order_seq_cst);

    checkCudaStatus(cudaStreamSynchronize(stream));

    evaluationWatch.setStopTick();        

    /// [ Alternative-Option ] cudaHostRegister ( void* ptr, size_t size, unsigned
    /// int  flags ) :: cudaHostRegisterMapped:

    /// ===============================================
    /// Aync Flow - wszystkie bloki za juz zaincjalizowane
    ///           - adressy blokow sa widoczne on the async-call-site [publikacja
    ///           adrressow/ arg capturing ]
    ///

    /// referencec
    // graph::Tensor A = graph::Tensor::fromDeviceMem(dev_A, N, N); /// Macierz
    // g�owna ukladu rownan liniowych graph::Tensor Fq; /// [size x size] Macierz
    // sztywnosci ukladu obiektow zawieszonych na sprezynach. graph::Tensor Wq;
    // /// [coffSize x size]  d(FI)/dq - Jacobian Wiezow

    /// HESSIAN
    // graph::Tensor Hs;

    // Wektor prawych stron [Fr; Fi]'
    // graph::Tensor SV = graph::Tensor::fromDeviceMem(dev_b, N, 1);

    // skladowe to Fr - oddzialywania pomiedzy obiektami, sily w sprezynach
    // graph::Tensor Fr = graph::Tensor::fromDeviceMem(dev_b, size , 1);

    // skladowa to Fiq - wartosci poszczegolnych wiezow
    // graph::Tensor Fi = graph::Tensor::fromDeviceMem(dev_b + size, coffSize, 1);

    stat->startTime = graph::ClockMillis()();

    printf("#=============== Solver Initialized =============# \n");
    printf("");

    /// LSM - ( reset )
    // linear_system_method_cuSolver_reset(stream);

    int itr = 0;

    /// #########################################################################
    ///
    /// idea - zapelniamy strumien w calym zakresie rozwiazania do CMAX -- i
    /// zczytujemy pierwszy mozliwy poprawny wynik
    ///
    while (itr < CMAX) {
        /// preinitialize data vector
        if (itr == 0) {
            /// SV - State Vector
            CopyIntoStateVector<<<ST_DIM_GRID, ST_DIM_BLOCK, Ns, stream>>>(dev_SV[0], d_points, size);

            /// SV -> setup Lagrange multipliers  -
            utility::memset(dev_SV[0] + size, 0, coffSize);
        } else {
            checkCudaStatus(
                cudaMemcpyAsync(dev_SV[itr], dev_SV[itr - 1], N * sizeof(double), cudaMemcpyDeviceToDevice, stream));
        }

        ///  Host Context - with references to device

        // computation data;
        ev[itr]->cID = itr;
        ev[itr]->SV = dev_SV[itr]; /// data vector lineage
        ev[itr]->A = dev_A;
        ev[itr]->b = dev_b;
        ev[itr]->dx = dev_dx;
        ev[itr]->dev_norm = dev_norm[itr];

        // geometric structure
        ev[itr]->size = size;
        ev[itr]->coffSize = coffSize;
        ev[itr]->dimension = dimension = N;

        ev[itr]->points = d_points;
        ev[itr]->geometrics = d_geometrics;
        ev[itr]->constraints = d_constraints;
        ev[itr]->parameters = d_parameters;

        ev[itr]->pointOffset = d_pointOffset;
        ev[itr]->constraintOffset = d_constraintOffset;
        ev[itr]->parameterOffset = d_parameterOffset;
        ev[itr]->accGeometricSize = d_accGeometricSize;
        ev[itr]->accConstraintSize = d_accConstraintSize;

        ///
        /// [ GPU ] computation context mapped onto devive object
        ///
        /// tu chce snapshot transfer

        utility::memcpyToDevice(&dev_ev[itr], ev[itr], 1);

        // #observation
        checkCudaStatus(cudaEventRecord(computeStart[itr], stream));

        // #observation
        checkCudaStatus(cudaEventRecord(prepStart[itr], stream));

        /// # KERNEL_PRE

        // if (itr > 0) {
        /// zerujemy macierz A      !!!!! second buffer
        utility::memset(dev_A, 0, N * N); // --- ze wzgledu na addytywnosc
                                          //}

        /// macierz `A
        /// Cooficients Stiffnes Matrix
        ComputeStiffnessMatrix<<<ST_DIM_GRID, ST_DIM_BLOCK, Ns, stream>>>(dev_ev[itr], geometrics.size());

        /// check state
        if (settings::get()->DEBUG_CHECK_ARG) {
            checkCudaStatus(cudaStreamSynchronize(stream));
            checkCudaStatus(cudaPeekAtLastError());
        }

        /// [ cuda / error ] 701 : cuda API failed 701,
        /// cudaErrorLaunchOutOfResources  = too many resources requested for launch
        //
        if (settings::get()->SOLVER_INC_HESSIAN) {
            EvaluateConstraintHessian<<<ST_DIM_GRID, ST_DIM_BLOCK, Ns, stream>>>(dev_ev[itr], constraints.size());
        }

        /// check state
        if (settings::get()->DEBUG_CHECK_ARG) {
            checkCudaStatus(cudaStreamSynchronize(stream));
            checkCudaStatus(cudaPeekAtLastError());
        }

        /*
            Lower Tensor Slice

            --- (Wq); /// Jq = d(Fi)/dq --- write without intermediary matrix
        */

        size_t dimBlock = constraints.size();
        EvaluateConstraintJacobian<<<ST_DIM_GRID, dimBlock, Ns, stream>>>(dev_ev[itr], constraints.size());

        /*
            Transposed Jacobian - Uperr Tensor Slice

            --- (Wq)'; /// Jq' = (d(Fi)/dq)' --- transposed - write without
           intermediary matrix
        */
        EvaluateConstraintTRJacobian<<<ST_DIM_GRID, dimBlock, Ns, stream>>>(dev_ev[itr], constraints.size());

        /// check state
        if (settings::get()->DEBUG_CHECK_ARG) {
            checkCudaStatus(cudaStreamSynchronize(stream));
            checkCudaStatus(cudaPeekAtLastError());
        }

        /// Tworzymy Macierz dest SV dest `b

        /// [ SV ]  - right hand site

        /// Fr /// Sily  - F(q) --  !!!!!!!
        EvaluateForceIntensity<<<ST_DIM_GRID, ST_DIM_BLOCK, Ns, stream>>>(dev_ev[itr], geometrics.size());

        /// check state
        if (settings::get()->DEBUG_CHECK_ARG) {
            checkCudaStatus(cudaStreamSynchronize(stream));
            checkCudaStatus(cudaPeekAtLastError());
        }

        /// Fi / Wiezy  - Fi(q)
        EvaluateConstraintValue<<<ST_DIM_GRID, ST_DIM_BLOCK, Ns, stream>>>(dev_ev[itr], constraints.size());

        /// check state
        if (settings::get()->DEBUG_CHECK_ARG) {
            checkCudaStatus(cudaStreamSynchronize(stream));
            checkCudaStatus(cudaPeekAtLastError());
        }

        /**
         * adresowanie jest tranzlatowane przy serializacji argumentow typu
         * PTR__device__
         *
         * Konwersja nie do zrealizowania long long  ptr = &*(dev_ev[itr])
         *
         * __device__ (ptr){
         *   A *a = () ptr
         * }
         */
        // printf("address %x  %p ,\ n", &dev_ev[itr]->A, &dev_ev[itr]->A);

        /// check state
        if (settings::get()->DEBUG_TENSOR_A) {
            stdoutTensorData<<<GRID_DBG, 1, Ns, stream>>>(dev_ev[itr], N, N);
            checkCudaStatus(cudaStreamSynchronize(stream));
        }

        if (settings::get()->DEBUG_TENSOR_B) {
            stdoutRightHandSide<<<GRID_DBG, 1, Ns, stream>>>(dev_ev[itr], N);
            checkCudaStatus(cudaStreamSynchronize(stream));
        }

        /// #observation
        checkCudaStatus(cudaEventRecord(prepStop[itr], stream));

        /// DENSE - CuSolver
        /// LU Solver
        ///

        /// ======== LINER SYSTEM equation CuSolver    === START

        /// #observation
        checkCudaStatus(cudaEventRecord(solverStart[itr], stream));

        linear_system_method_cuSolver(dev_A, dev_b, N, stream);

        /// #observation
        checkCudaStatus(cudaEventRecord(solverStop[itr], stream));

        /// ======== LINER SYSTEM equation CuSolver    === STOP

        /// uaktualniamy punkty [ SV ] = [ SV ] + [ delta ]

        StateVectorAddDifference<<<DIM_GRID, DIM_BLOCK, Ns, stream>>>(dev_SV[itr], dev_b, N);

        // print actual state vector single kernel

        if (settings::get()->DEBUG_TENSOR_SV) {
            stdoutStateVector<<<GRID_DBG, 1, Ns, stream>>>(dev_ev[itr], N);
            checkCudaStatus(cudaStreamSynchronize(stream));
        }

        /// check state
        if (settings::get()->DEBUG_CHECK_ARG) {
            checkCudaStatus(cudaStreamSynchronize(stream));
            checkCudaStatus(cudaPeekAtLastError());
        }

        /// --- KERNEL_POST( __synchronize__ REDUCTION )

        /// write into __device__ Computation block

        ///  uzupelnic #Question: Vector B = Fi(q) = 0 przeliczamy jeszce raz !!!


        // synchronize -- "adress still const"
        /// - not used
        utility::memcpyFromDevice(ev[itr], dev_ev[itr], 1);


        double *host_norm = &ev[itr]->norm;

        ConstraintGetFullNorm(coffSize, size, dev_b, host_norm, stream);
       

        /// ============================================================
        ///
        ///         copy DeviceComputation to HostComputation -- addCallback
        ///
        /// ============================================================

        /// retrive ComputationFrom device

        void *userData = static_cast<void *>(ev[itr]); // local_Computation

        // typedef void (CUDART_CB *cudaStreamCallback_t)(cudaStream_t stream,
        // cudaError_t status, void *userData);

        /// callback dispatched to special cuda p_thread
        checkCudaStatus(cudaStreamAddCallback(stream, computationResultHandler, userData, 0));

        checkCudaStatus(cudaEventRecord(computeStop[itr], stream));

        itr++;
    }

    /// all computation tiles submited - for first offset , or invalid last

    std::unique_lock<std::mutex> ulock(mutex);

    /// atomic read
    if (result.load(std::memory_order_seq_cst) == nullptr) {
        /// spurious wakeup
        condition.wait(ulock, [&] { return result.load(std::memory_order_seq_cst) != nullptr; });
    }

    ComputationState *computation;

    computation = result.load(std::memory_order_seq_cst);

    solverWatch.setStopTick();

    // condition met
    ///
    ///
    ///
    ///
    /// HOST computation view

    // STATE VECTOR offset

    // oczekuje strumienia ale nie kopiujemy danych

    checkCudaStatus(cudaStreamSynchronize(stream));

    if (settings::get()->DEBUG) {

        long long solutionDelta = solverWatch.delta();

        printf("\n");
        printf("===================================================\n");
        printf("\n");
        printf("solution time delta [ns]        ( %zd )\n", solutionDelta);
        printf("\n");
        printf("===================================================\n");
        printf("comp ID                         ( %d )\n", computation->cID);
        printf("computation norm                ( %e )\n", computation->norm);
        printf("\n");
        printf("computation size                ( %zd )\n", computation->size);
        printf("computation coffSize            ( %zd )\n", computation->coffSize);
        printf("computation dimension           ( %zd )\n", computation->dimension);
        printf("\n");
        printf("===================================================\n");
        printf("\n");
    }

    CopyFromStateVector<<<DIM_GRID, DIM_BLOCK, Ns, stream>>>(d_points, computation->SV, size);

    utility::memcpyFromDevice(points, d_points);

    double SOLVER_EPSILON = (settings::get()->CU_SOLVER_EPSILON);

    stat->startTime = solverWatch.getStartTick();
    stat->stopTime = solverWatch.getStopTick();
    stat->timeDelta = solverWatch.delta();       
    
    stat->size = size;
    stat->coefficientArity = coffSize;
    stat->dimension = dimension;


    stat->accEvaluationTime = evaluationWatch.delta();  /// !! nasz wewnetrzny allocator pamieci !
    stat->accSolverTime = solverWatch.delta();

    stat->convergence = computation->norm < SOLVER_EPSILON;
    stat->error = computation->norm;
    stat->constraintDelta = computation->norm;
    stat->iterations = computation->cID;


    /// w drugiej fazie dopisac => dodatkowo potrzebny per block  __shared__
    /// memory   ORR    L1   fast region memory
    ///
    /// Evaluation data for  device  - CONST DATE for in process execution

    checkCudaStatus(cudaStreamSynchronize(stream));

    goto Error;

/// free allocated memory in solution
Error:

    *error = cudaSuccess;
}

} // namespace solver

/// @brief Setup __device__ dest of geometric points in this moment.
///
/// @param ec evaluation context
/// @param N size of geometric object data dest
/// @param _point[] __shared__ reference into model point data
/// @tparam TZ tensor dimension without constraints
/// @return void
///
