#include "geometric_solver.h"

#include "cuda_runtime.h"
#include "stop_watch.h"

#include <atomic>
#include <condition_variable>
#include <memory>

#include <functional>
#include <numeric>

#include <stdexcept>
#include <stdio.h>

#include "model.cuh"
#include "solver_kernel.cuh"

#include "linear_system.h"

#include "model_config.h"

/// MAX SOLVER ITERATIONS
#define CMAX 20

#define CONVERGENCE_LIMIT 10e-5

/// ------------ domain model przenosimy do pliku *.cu i laczymy *.cuh -> do *.cu

/// points register
static std::vector<graph::Point> points; /// poLocations id-> point_offset

/// geometricc register
static std::vector<graph::Geometric> geometrics; /// ==> Macierz A, accumulative offset for each primitive

/// constraints register
static std::vector<graph::Constraint> constraints; /// ===> Wiezy , accumulative offset for each constraint

/// parameters register
static std::vector<graph::Parameter> parameters; /// paramLocation id-> param_offset

/// Point  Offset in computation matrix [id] -> point offset   ~~ Gather Vectors
static std::unique_ptr<int[]> pointOffset;

/// Parameter Offset mapper from [id] -> parameter offset in reference dest
static std::unique_ptr<int[]> parameterOffset;

/// Accymulated Geometric Object Size
static std::unique_ptr<int[]> accGeometricSize; /// 2 * point.size()

/// Accumulated Constraint Size
static std::unique_ptr<int[]> accConstraintSize;

/// === Solver Performance Watchers

/// observation of submited tasks
static graph::StopWatch solverWatch;

/// observation of computation time - single computation run
static cudaEvent_t computeStart[CMAX] = {nullptr};
static cudaEvent_t computeStop[CMAX] = {nullptr};

/// observation of matrices  preperations
static cudaEvent_t prepStart[CMAX] = {nullptr};
static cudaEvent_t prepStop[CMAX] = {nullptr};

/// observation of accumalated cuSolver method
static cudaEvent_t solverStart[CMAX] = {nullptr};
static cudaEvent_t solverStop[CMAX] = {nullptr};

static int size;      /// wektor stanu
static int coffSize;  /// wspolczynniki Lagrange
static int dimension; /// dimension = size + coffSize

/// ===========================================================
/// Async Computation - Implicit ref in utility:: namespace
///
static cudaStream_t stream = NULL;

// --- inflow, --- graph / sekwencja zadan
// --- outflow --- zdejmujemy kopiowanie

///
/// ===========================================================

namespace utility
{

template <typename Ty> void mallocHost(Ty **dest, size_t size)
{
    checkCudaStatus(cudaMallocHost((void **)dest, size * sizeof(Ty), cudaHostAllocDefault));
    // * - ::cudaHostAllocMapped: Maps the allocation into the CUDA address space.
    // * - - The device pointer to the memory may be obtained by calling * ::cudaHostGetDevicePointer()
}

template <typename Ty> void mallocAsync(Ty **dest, size_t size)
{
    checkCudaStatus(cudaMallocAsync((void **)dest, size * sizeof(Ty), stream));
}

template <typename Ty> void memcpyToDevice(Ty **dest_device, const Ty *const &vector, size_t size)
{
    /// transfer into new allocation host_vector
    checkCudaStatus(cudaMemcpyAsync(*dest_device, vector, size * sizeof(Ty), cudaMemcpyHostToDevice, stream));
}

template <typename Ty> void memcpyToDevice(Ty **dest_device, std::vector<Ty> const &vector)
{
    /// memcpy to device
    memcpyToDevice(dest_device, vector.data(), vector.size());
}

template <typename Ty> void mallocToDevice(Ty **dev_ptr, size_t size)
{
    /// safe malloc
    checkCudaStatus(cudaMallocAsync((void **)dev_ptr, size, stream));
}

template <typename Ty> void memcpyFromDevice(std::vector<Ty> &vector, Ty *src_device)
{
    /// transfer into new allocation host_vector
    checkCudaStatus(
        cudaMemcpyAsync(vector.data(), src_device, vector.size() * sizeof(Ty), cudaMemcpyDeviceToHost, stream));
}

template <typename Ty> void memcpyFromDevice(Ty *dest, Ty *src_device, size_t arity)
{
    /// transfer into new allocation
    checkCudaStatus(cudaMemcpyAsync(dest, src_device, arity * sizeof(Ty), cudaMemcpyDeviceToHost, stream));
}

template <typename Ty> void freeMem(Ty *dev_ptr)
{
    /// safe free mem
    checkCudaStatus(cudaFreeAsync(dev_ptr, stream));
}

template <typename Ty> void freeHostMem(Ty *ptr)
{
    /// safe free mem
    checkCudaStatus(cudaFreeHost(ptr));
}

template <typename Ty> void memset(Ty *dev_ptr, int value, size_t size)
{
    ///
    checkCudaStatus(cudaMemsetAsync(dev_ptr, value, size * sizeof(Ty), stream));
}

template <typename Obj, typename ObjIdFunction>
std::unique_ptr<int[]> stateOffset(std::vector<Obj> objects, ObjIdFunction objectIdFunction)
{
    if (objects.empty())
    {
        return std::unique_ptr<int[]>(new int[0]);
    }

    std::unique_ptr<int[]> offsets(new int[objectIdFunction(++objects.rbegin())]);
    auto iterator = objects.begin();
    int offset = 0;
    while (iterator != objects.end())
    {
        auto objectId = objectIdFunction(iterator);
        offsets[objectId] = offset++;
        iterator++;
    }
    return offsets;
}

/// #/include <numeric> std::partial_sum

template <typename Obj, typename ValueFunction>
std::unique_ptr<int[]> accumalatedValue(std::vector<Obj> vector, ValueFunction valueFunction)
{
    int accValue = 0;
    std::unique_ptr<int[]> accumulated(new int[vector.size()]);
    for (int offset = 0; offset < vector.size(); offset++)
    {
        accumulated[offset] = accValue;
        accValue = accValue + valueFunction(vector[offset]);
    }
    return accumulated;
}

} // namespace utility

namespace solver
{

void resetComputationData(cudaError_t *error)
{
    std::remove_if(points.begin(), points.end(), [](auto _) { return true; });
    std::remove_if(geometrics.begin(), geometrics.end(), [](auto _) { return true; });
    std::remove_if(constraints.begin(), constraints.end(), [](auto _) { return true; });
    std::remove_if(parameters.begin(), parameters.end(), [](auto _) { return true; });

    pointOffset = NULL;
    parameterOffset = NULL;
    accConstraintSize = NULL;
    accGeometricSize = NULL;
}

/**
 *
 */
void resetComputationContext(cudaError_t *error)
{
}

/**
 *
 */
void initComputationContext(cudaError_t *error)
{
    if (points.size() == 0)
    {
        throw new std::exception("empty solution space, add some geometric types");
    }

    /// mapping from point Id => point dest offset
    pointOffset = utility::stateOffset(points, [](auto point) { return point->id; });
    /// this is mapping from Id => parameter dest offset
    parameterOffset = utility::stateOffset(parameters, [](auto parameter) { return parameter->id; });
    /// accumalted position of geometric block
    accGeometricSize = utility::accumalatedValue(geometrics, graph::geometricSetSize);
    /// accumulated position of constrain block
    accConstraintSize = utility::accumalatedValue(constraints, graph::constraintSize);

    solverWatch.reset();

    /// `A` tensor internal structure dimensions
    size = std::accumulate(geometrics.begin(), geometrics.end(), 0,
                           [](auto acc, auto const &geometric) { return acc + graph::geometricSetSize(geometric); });
    coffSize = std::accumulate(constraints.begin(), constraints.end(), 0, [](auto acc, auto const &constraint) {
        return acc + graph::constraintSize(constraint);
    });

    dimension = size + coffSize;

    /**
     * po zarejestrowaniu calego modelu w odpowiadajacych rejestrach , zainicjalizowac pomocnicze macierze
     *
     *
     *
     * przeliczenie pozycji absolutnej punktu na macierzy wyjsciowej
     */
    if (stream == NULL)
    {
        // implicit in utility::
        checkCudaStatus(cudaStreamCreate(&stream));

        int evID = 0;
        while (evID < CMAX)
        {
            // #observations
            checkCudaStatus(cudaEventCreate(&prepStart[evID]));
            checkCudaStatus(cudaEventCreate(&prepStop[evID]));
            checkCudaStatus(cudaEventCreate(&computeStart[evID]));
            checkCudaStatus(cudaEventCreate(&computeStop[evID]));
            checkCudaStatus(cudaEventCreate(&solverStart[evID]));
            checkCudaStatus(cudaEventCreate(&solverStop[evID]));

            evID++;
        }
    }
}

/**
 *
 */
void registerPointType(int id, double px, double py)
{
    points.emplace_back(id, px, py);
}

/**
 *
 */
void registerGeometricType(int id, int geometricTypeId, int p1, int p2, int p3, int a, int b, int c, int d)
{
    geometrics.emplace_back(id, geometricTypeId, p1, p2, p3, a, b, c, d);
}

/**
 *
 */
void registerParameterType(int id, double value)
{
    parameters.emplace_back(id, value);
}

/**
 *
 */
void registerConstraintType(int id, int jconstraintTypeId, int k, int l, int m, int n, int paramId, double vecX,
                            double vecY)
{
    constraints.emplace_back(id, jconstraintTypeId, k, l, m, n, paramId, vecX, vecY);
}

/**
 *
 */
double getPointPXCoordinate(int id)
{
    int offset = pointOffset[id];
    double px = points[offset].x;
    return px;
}

/**
 *
 */
double getPointPYCoordinate(int id)
{
    int offset = pointOffset[id];
    double py = points[offset].y;
    return py;
}

void getPointCoordinateVector(double *state_arr)
{
}

/// <summary>
///
/// </summary>
/// <param name="ev">[__host__]</param>
/// <param name="offset">[__device__]</param>
/// <param name="stream"></param>
void ConstraintGetFullNorm(size_t coffSize, size_t size, double *b, double *result, cudaStream_t stream)
{
    linear_system_method_cuBlas_vectorNorm(coffSize, (b + size), result, stream);
}

__host__ __device__ double *getComputationNormFieldOffset(ComputationState *dev_ev)
{
    return &dev_ev->norm;
}

/// D.3.1.1. Device-Side Kernel Launch - kernel default shared memory, number of bytes
constexpr size_t Ns = 0;

/// grid size
constexpr size_t DIM_GRID = 1;

/*
   The maximum registers per thread is 255.

    CUDAdrv.MAX_THREADS_PER_BLOCK, which is good, ( 1024 )

    " if your kernel uses many registers, it also limits the amount of threads you can use."

*/

/// https://docs.nvidia.com/cuda/turing-tuning-guide/index.html#sm-occupancy
///
/// thread block size
constexpr size_t DIM_BLOCK = 512;

/// lock for escaped data from computation rail
/// -- first conveged computation contex or last invalid

/// shared with cuda stream callback for wait, notify mechanism
std::condition_variable condition;

std::mutex mutex;

// host reference guarded by mutex
std::atomic<ComputationState *> result;

/// <summary>
/// Computation Round Handler
/// </summary>
/// <param name="userData"></param>
void computationResultHandler(cudaStream_t stream, cudaError_t status, void *userData)
{
    // synchronize on stream

    ComputationState *computation = static_cast<ComputationState *>(userData);

#ifdef CDEBUG
    printf("handler: %d \n", computation->cID);
#endif CDEBUG

    // obsluga bledow w strumieniu
    if (status != cudaSuccess)
    {
        const char *errorName = cudaGetErrorName(status);
        const char *errorStr = cudaGetErrorString(status);

        printf("[error] - computation id [%d] ,  %s = $s \n", computation->cID, errorName, errorStr);
        return;
    }

    bool last = computation->cID == (CMAX - 1);

    if (computation->norm < CONVERGENCE_LIMIT || last)
    {
        /// update offset
        result.store(computation, std::memory_order_seq_cst);

        condition.notify_one();
    }

    // synchronize with stream next computation
}

__device__ constexpr const char *WIDEN_DOUBLE_STR_FORMAT = "%26d";

__device__ constexpr const char *FORMAT_STR_DOUBLE = " %11.2e";
__device__ constexpr const char *FORMAT_STR_DOUBLE_CM = ", %11.2e";


__global__ void stdoutTensorData(ComputationState *ev, size_t ld, size_t cols)
{
    const graph::Tensor t = graph::Tensor::fromDeviceMem(ev->A, ld, cols);

    printf("A ,,,\ n");
    printf("\n MatrixDouble - %d x %d ****************************************\n", t.ld, t.cols);

    /// table header
    for (int i = 0; i < cols / 2; i++)
    {
        printf(WIDEN_DOUBLE_STR_FORMAT, i);
    }
    printf("\n");

    /// table data

    for (int i = 0; i < ld; i++)
    {
        printf(FORMAT_STR_DOUBLE, t.getValue(i, 0));

        for (int j = 1; j < cols; j++)
        {
            printf(FORMAT_STR_DOUBLE_CM, t.getValue(i, j));
        }
        if (i < cols - 1)
            printf("\n");
    }
}

__global__ void stdoutRightHandSide(ComputationState *ev, size_t ld)
{
    const graph::Tensor b = graph::Tensor::fromDeviceMem(ev->b, ld, 1);

    printf("\n b \n");
    printf("\n MatrixDouble - %d x %d ****************************************\n", b.ld, b.cols);
    printf("\n");
    /// table data

    for (int i = 0; i < ld; i++)
    {
        printf(FORMAT_STR_DOUBLE, b.getValue(i, 0));
        printf("\n");
    }

    printf("\n\n");
}

#undef CDEBUG

///
/// Setup all matricies for computation and prepare kernel stream  intertwined with cuSolver
///
///
void solveSystemOnGPU(solver::SolverStat *stat, cudaError_t *error)
{
    /// # Consideration -> ::"ingest stream and observe" fetch first converged
    ///
    ///
    ///   -- "cublas_v2.h" funkcje API czytaj/i wpisuja  wspolczynnik z HOST jak i z DEVICE
    ///
    ///   -- zalaczamy zadanie FI(q) = 0 norm -> wpiszemy do ExecutionContext variable
    ///
    ///   --- state-dest lineage memcpy(deviceToDevice)
    ///

    int N = dimension;

    /// prepare local offset context
    result.store(NULL, std::memory_order_seq_cst);


    checkCudaStatus(cudaDeviceSynchronize());


    /// Uklad rownan liniowych  [ A * x = b ] powsta³y z linerazycji ukladu dynamicznego - tensory na urzadzeniu.
    double *dev_A = nullptr;
    double *dev_b = nullptr;
    double *dev_dx = nullptr;      /// [ A ] * [ dx ] = [ b ]
    double *dev_SV[CMAX] = {NULL}; /// STATE VECTOR  -- lineage

    ComputationState *dev_ev[CMAX] = {NULL}; /// device reference

    /// Local Computation References
    ComputationState *ev[CMAX] = {NULL};

    /// w drugiej fazie dopisac => dodatkowo potrzebny per block  __shared__  memory   ORR    L1   fast region memory
    ///
    /// Evaluation data for  device  - CONST DATE for in process execution

    graph::Point *d_points = nullptr;
    graph::Geometric *d_geometrics = nullptr;
    graph::Constraint *d_constraints = nullptr;
    graph::Parameter *d_parameters = nullptr;

    int *d_pointOffset;       /// ---
    int *d_parameterOffset;   ///
    int *d_accGeometricSize;  /// accumulative offset with geometric size evaluation function
    int *d_accConstraintSize; /// accumulative offset with constraint size evaluation function

    /// model aggreate for const immutable data

    if (points.size() == 0)
    {
        printf("[solver] - empty evaluation space model\n");
        return;
    }

    if (constraints.size() == 0)
    {
        printf("[solver] - no constraint configuration applied onto model\n");
        return;
    }

    /// cuSolver component configuration
    int major = 0;
    int minor = 0;
    int patch = 0;

    checkCuSolverStatus(cusolverGetProperty(MAJOR_VERSION, &major));
    checkCuSolverStatus(cusolverGetProperty(MINOR_VERSION, &minor));
    checkCuSolverStatus(cusolverGetProperty(PATCH_LEVEL, &patch));
    printf("[ CUSOLVER ]  version ( Major.Minor.PatchLevel): %d.%d.%d\n", major, minor, patch);

    /// for [GPU] Evaluation  - dane obliczeniowego modelu !
    ///
    /// ============================================================
    ///
    ///         Host Computation with references to device
    ///
    /// ============================================================

    /// const data in computation
    utility::mallocAsync(&d_points, points.size());
    utility::mallocAsync(&d_geometrics, 2 /**  geometrics.size() */);
    utility::mallocAsync(&d_constraints, constraints.size());
    utility::mallocAsync(&d_parameters, parameters.size());

    size_t pointOffsetSz = (points.rbegin())->id + 1;
    utility::mallocAsync(&d_pointOffset, pointOffsetSz);

    size_t parameterOffsetSz = (parameters.empty()) ? 0 : (parameters.rbegin())->id + 1;
    utility::mallocAsync(&d_parameterOffset, parameterOffsetSz);
    utility::mallocAsync(&d_accGeometricSize, geometrics.size());
    utility::mallocAsync(&d_accConstraintSize, constraints.size());

    // immutables -
    utility::memcpyToDevice(&d_points, points);
    utility::memcpyToDevice(&d_geometrics, geometrics);
    utility::memcpyToDevice(&d_constraints, constraints);
    utility::memcpyToDevice(&d_parameters, parameters);

    // immutables
    utility::memcpyToDevice(&d_pointOffset, pointOffset.get(), pointOffsetSz);
    utility::memcpyToDevice(&d_parameterOffset, parameterOffset.get(), parameterOffsetSz);
    utility::memcpyToDevice(&d_accGeometricSize, accGeometricSize.get(), geometrics.size());
    utility::memcpyToDevice(&d_accConstraintSize, accConstraintSize.get(), constraints.size());

    ///
    ///  [ GPU ] tensors `A` `x` `dx` `b`  ------------------------
    ///
    /// one allocation block - slice
    //
    utility::mallocAsync(&dev_A, N * N);
    utility::mallocAsync(&dev_b, N);
    utility::mallocAsync(&dev_dx, N);

    for (int itr = 0; itr < CMAX; ++itr)
    {

        utility::mallocAsync(&dev_SV[itr], N); /// each computation state with its own StateVector
        utility::mallocAsync(&dev_ev[itr], 1); /// each computation state with its own device Evalution Context
    }

    checkCudaStatus(cudaStreamSynchronize(stream));

    /// [ Alternative-Option ] cudaHostRegister ( void* ptr, size_t size, unsigned int  flags ) ::
    /// cudaHostRegisterMapped:

    for (int itr = 0; itr < CMAX; ++itr)
    {
        utility::mallocHost(&ev[itr], 1); /// each computation state with its own host Evalution Context
    }

    /// ===============================================
    /// Aync Flow - wszystkie bloki za juz zaincjalizowane
    ///           - adressy blokow sa widoczne on the async-call-site [publikacja adrressow/ arg capturing ]
    ///

    /// referencec
    // graph::Tensor A = graph::Tensor::fromDeviceMem(dev_A, N, N); /// Macierz g³owna ukladu rownan liniowych
    // graph::Tensor Fq; /// [size x size]     Macierz sztywnosci ukladu obiektow zawieszonych na sprezynach.
    // graph::Tensor Wq; /// [coffSize x size]  d(FI)/dq - Jacobian Wiezow

    /// HESSIAN
    // graph::Tensor Hs;

    // Wektor prawych stron [Fr; Fi]'
    // graph::Tensor b = graph::Tensor::fromDeviceMem(dev_b, N, 1);

    // skladowe to Fr - oddzialywania pomiedzy obiektami, sily w sprezynach
    // graph::Tensor Fr = graph::Tensor::fromDeviceMem(dev_b, size , 1);

    // skladowa to Fiq - wartosci poszczegolnych wiezow
    // graph::Tensor Fi = graph::Tensor::fromDeviceMem(dev_b + size, coffSize, 1);

    stat->startTime = graph::TimeNanosecondsNow();

    printf("#=============== Solver Initialized =============# \n");
    printf("");

    /// LSM - ( reset )
    linear_system_method_cuSolver_reset(stream);

    int itr = 0;

    /// #########################################################################
    ///
    /// idea - zapelniamy strumien w calym zakresie rozwiazania do CMAX -- i zczytujemy pierwszy mozliwy poprawny wynik
    ///
    while (itr < CMAX)
    {
        /// preinitialize state vector
        if (itr == 0)
        {
            /// SV - State Vector
            CopyIntoStateVector<<<DIM_GRID, DIM_BLOCK, Ns, stream>>>(dev_SV[0], d_points, size);

            /// SV -> setup Lagrange multipliers  -
            utility::memset(dev_SV[0] + size, 0, coffSize);
        }
        else
        {
            checkCudaStatus(cudaMemcpyAsync(dev_SV[itr], dev_SV[itr - 1], N * sizeof(double), cudaMemcpyDeviceToDevice, stream));
        }

        ///  Host Context - with references to device

        // computation state;
        ev[itr]->cID = itr;
        ev[itr]->SV = dev_SV[itr]; /// state vector lineage
        ev[itr]->A = dev_A;
        ev[itr]->b = dev_b;
        ev[itr]->dx = dev_dx;

        // geometric structure
        ev[itr]->size = size;
        ev[itr]->coffSize = coffSize;
        ev[itr]->dimension = dimension = N;

        ev[itr]->points = d_points;
        ev[itr]->geometrics = d_geometrics;
        ev[itr]->constraints = d_constraints;
        ev[itr]->parameters = d_parameters;

        ev[itr]->pointOffset = d_pointOffset;
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

        if (itr > 1)
        {
            /// zerujemy macierz A      !!!!! second buffer
            utility::memset(dev_A, 0, N * N); // --- ze wzgledu na addytywnosc
        }

        /// macierz `A
        /// Cooficients Stiffnes Matrix
        ComputeStiffnessMatrix<<<DIM_GRID, DIM_BLOCK, Ns, stream>>>(dev_ev[itr], geometrics.size());

#ifdef CDEBUG
        checkCudaStatus(cudaStreamSynchronize(stream));
        checkCudaStatus(cudaPeekAtLastError());
#endif

        /// [ cuda / error ] 701 : cuda API failed 701,  cudaErrorLaunchOutOfResources  = too many resources requested
        /// for launch
        EvaluateConstraintHessian<<<DIM_GRID, DIM_BLOCK, Ns, stream>>>(dev_ev[itr], constraints.size());

        /// ---?----symmetric---?----

#ifdef CDEBUG
        checkCudaStatus(cudaStreamSynchronize(stream));
        checkCudaStatus(cudaPeekAtLastError());
#endif

        /// upper Triangular
        /// --- (Wq); /// Jq = d(Fi)/dq --- write without intermediary matrix   `A = `A set value

        EvaluateConstraintJacobian<<<DIM_GRID, DIM_BLOCK, Ns, stream>>>(dev_ev[itr], constraints.size());

#ifdef CDEBUG
        checkCudaStatus(cudaStreamSynchronize(stream));
        checkCudaStatus(cudaPeekAtLastError());
#endif

        /// Tworzymy Macierz dest b dest `b

        /// [ b ]  - right hand site

        /// Fr /// Sily  - F(q) --  .mulitply(-1);
        EvaluateForceIntensity<<<DIM_GRID, DIM_BLOCK, Ns, stream>>>(dev_ev[itr], geometrics.size());

#ifdef CDEBUG
        checkCudaStatus(cudaStreamSynchronize(stream));
        checkCudaStatus(cudaPeekAtLastError());
#endif

        /// Fi / Wiezy  - Fi(q)  --  .mulitply(-1);
        EvaluateConstraintValue<<<DIM_GRID, DIM_BLOCK, Ns, stream>>>(dev_ev[itr], constraints.size());

#ifdef CDEBUG
        checkCudaStatus(cudaStreamSynchronize(stream));
        checkCudaStatus(cudaPeekAtLastError());
#endif

        /**
         * adresowanie jest tranzlatowane przy serializacji argumentow typu PTR__device__
         *
         * Konwersja nie do zrealizowania long long  ptr = &*(dev_ev[itr])
         *
         * __device__ (ptr){
         *   A *a = () ptr
         * }
         */
        // printf("address %x  %p ,\ n", &dev_ev[itr]->A, &dev_ev[itr]->A);

        stdoutTensorData<<<DIM_GRID, 1, Ns, stream>>>(dev_ev[itr], N, N);
        stdoutRightHandSide<<<DIM_GRID, 1, Ns, stream>>>(dev_ev[itr], N);
        checkCudaStatus(cudaStreamSynchronize(stream));

        ///  upper traingular

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

#ifdef CDEBUG
        checkCudaStatus(cudaStreamSynchronize(stream));
        checkCudaStatus(cudaPeekAtLastError());
#endif

        /// --- KERNEL_POST( __synchronize__ REDUCTION )

        /// write into __device__ Computation block

        ///  uzupelnic #Question: Vector B = Fi(q) = 0 przeliczamy jeszce raz !!!

        double *dev_norm = getComputationNormFieldOffset(dev_ev[itr]);

        ConstraintGetFullNorm(coffSize, size, dev_b, dev_norm, stream);

        // synchronize -- "adress still const"
        utility::memcpyFromDevice(ev[itr], dev_ev[itr], 1);

        /// ============================================================
        ///
        ///         copy DeviceComputation to HostComputation -- addCallback
        ///
        /// ============================================================

        /// retrive ComputationFrom device

        void *userData = static_cast<void *>(ev[itr]); // local_Computation

        /// callback dispatched to special cuda p_thread
        checkCudaStatus(cudaStreamAddCallback(stream, computationResultHandler, userData, 0));

        checkCudaStatus(cudaEventRecord(computeStop[itr], stream));

        itr++;
    }

    /// all computation tiles submited - for first offset , or invalid last

    std::unique_lock<std::mutex> ulock(mutex);

    /// atomic read
    if (result.load(std::memory_order_seq_cst) == nullptr)
    {
        /// spurious wakeup
        condition.wait(ulock, [] { return result.load(std::memory_order_seq_cst) != nullptr; });
    }

    // condition met
    ///
    ///
    ///
    ///
    /// HOST computation view
    ComputationState *computation = result.load(std::memory_order_seq_cst);

    // STATE VECTOR offset

    // oczekuje strumienia ale nie kopiujemy danych

    checkCudaStatus(cudaStreamSynchronize(stream));

    solverWatch.setStopTick();
    long solutionDelta = solverWatch.delta();

    printf("# solution delta : %d \n", solutionDelta); // print execution time
    printf("\n");                                      // print execution time

    CopyFromStateVector<<<DIM_GRID, DIM_BLOCK, Ns, stream>>>(d_points, computation->SV, size);

    utility::memcpyFromDevice(points, d_points);

    stat->constraintDelta = computation->norm;
    stat->convergence = computation->norm < CONVERGENCE_LIMIT;
    stat->stopTime = graph::TimeNanosecondsNow();
    stat->iterations = computation->cID;
    stat->accSolverTime = 0;
    stat->accEvaluationTime = 0;
    stat->timeDelta = solverWatch.stopTick - solverWatch.startTick;

    /// w drugiej fazie dopisac => dodatkowo potrzebny per block  __shared__  memory   ORR    L1   fast region memory
    ///
    /// Evaluation data for  device  - CONST DATE for in process execution

    checkCudaStatus(cudaStreamSynchronize(stream));

/// free allocated memory in solution
Error:
    utility::freeHostMem(ev[0]);

    utility::freeMem(dev_ev[0]);
    utility::freeMem(dev_A);
    utility::freeMem(dev_b);
    utility::freeMem(dev_SV[0]);
    utility::freeMem(dev_dx);

    utility::freeMem(d_points);
    utility::freeMem(d_geometrics);
    utility::freeMem(d_constraints);
    utility::freeMem(d_parameters);

    utility::freeMem(d_pointOffset);       /// point offset table
    utility::freeMem(d_accGeometricSize);  /// accumulative offset with geometric size evaluation function
    utility::freeMem(d_accConstraintSize); /// accumulative offset with constraint size evaluation function

    checkCudaStatus(cudaStreamSynchronize(stream));
    // implicit object for utility
    if (stream != NULL)
    {
        checkCudaStatus(cudaStreamDestroy(stream));
    }

    *error = cudaSuccess;
}

} // namespace solver

/// KERNEL#
void ConstraintEvaluateConstraintVector()
{
    /// Wiezy  - Fi(q)
    /// b.mulitply(-1);
}

/// @brief Setup __device__ dest of geometric points in this moment.
///
/// @param ec evaluation context
/// @param N size of geometric object data dest
/// @param _point[] __shared__ reference into model point state
/// @tparam TZ tensor dimension without constraints
/// @return void
///

/// <summary>
///
/// </summary>
/// <param name="status"></param>
/// <param name="__line_"></param>
void checkCudaStatus_impl(cudaError_t status, size_t __line_)
{
    if (status != cudaSuccess)
    {
        printf("%d: cuda API failed with status %d\n", static_cast<int>(__line_), static_cast<int>(status));
        throw std::logic_error("cuda API error");
    }
}