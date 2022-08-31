#include "gpu_computation.h"

#include "cuda_runtime_api.h"

#include <typeinfo>
#include <numeric>
#include <functional>

#include "utility.cuh"
#include "solver_kernel.cuh"
#include "linear_system.h"

namespace solver {

/// ====================

GPUComputation::GPUComputation(long computationId, std::shared_ptr<GPUComputationContext> _cc,
                               std::vector<graph::Point> &&_points, std::vector<graph::Geometric> &&_geometrics,
                               std::vector<graph::Constraint> &&_constraints, std::vector<graph::Parameter> &&_parameters)
    : computationId(computationId), cc(_cc), stream(_cc->get_stream()), points(std::move(_points)),
      geometrics(std::move(_geometrics)), constraints(std::move(_constraints)), parameters(std::move(_parameters)) {
    if (points.size() == 0) {
        printf("empty solution space, add some geometric types\n");
        *error = cudaSuccess;
        return;
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

    /// setup all dependent structure for device , accSize or accOffset
    preInitializeData();

    /// prepera allocation and transfer structures onto device
    memcpyComputationToDevice();

    printf("[solver] model stage consistent !\n");
}

//////////////////===================================

void GPUComputation::preInitializeData() {

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
}

void GPUComputation::memcpyComputationToDevice() {

    /// ============================================================
    ///         Host Computation with references to Device
    /// ============================================================

    /// const data in computation
    utility::mallocAsync(&d_points, points.size(), stream);
    utility::mallocAsync(&d_geometrics, geometrics.size(), stream);
    utility::mallocAsync(&d_constraints, constraints.size(), stream);
    utility::mallocAsync(&d_parameters, parameters.size(), stream);

    utility::mallocAsync(&d_pointOffset, pointOffset.size(), stream);
    utility::mallocAsync(&d_constraintOffset, constraintOffset.size(), stream);
    utility::mallocAsync(&d_parameterOffset, parameterOffset.size(), stream);
    utility::mallocAsync(&d_accGeometricSize, geometrics.size(), stream);
    utility::mallocAsync(&d_accConstraintSize, constraints.size(), stream);

    // immutables -
    utility::memcpyAsync(&d_points, points, stream);
    utility::memcpyAsync(&d_geometrics, geometrics, stream);
    utility::memcpyAsync(&d_constraints, constraints, stream);
    utility::memcpyAsync(&d_parameters, parameters, stream);

    // immutables
    utility::memcpyAsync(&d_pointOffset, pointOffset.data(), pointOffset.size(), stream);
    utility::memcpyAsync(&d_constraintOffset, constraintOffset.data(), constraintOffset.size(), stream);

    utility::memcpyAsync(&d_accGeometricSize, accGeometricSize.data(), accGeometricSize.size(), stream);
    utility::memcpyAsync(&d_accConstraintSize, accConstraintSize.data(), accConstraintSize.size(), stream);
    if (!parameters.empty()) {
        utility::memcpyAsync(&d_parameterOffset, parameterOffset.data(), parameterOffset.size(), stream);
    }

    size_t N = dimension;
    ///
    ///  [ GPU ] tensors `A` `x` `dx` `b`  ------------------------
    ///
    utility::mallocAsync(&dev_A, N * N, stream);
    utility::mallocAsync(&dev_b, N, stream);
    utility::mallocAsync(&dev_dx, N, stream);

    for (int itr = 0; itr < CMAX; itr++) {
        /// each computation data with its own StateVector
        utility::mallocAsync(&dev_SV[itr], N, stream);
    }
}

// std::function/ std::bind  does not provide reference to raw C pointer
static GPUComputation *_registerComputation;

void GPUComputation::computationResultHandlerDelegate(cudaStream_t stream, cudaError_t status, void *userData) {
    if (_registerComputation) {
        _registerComputation->computationResultHandler(stream, status, userData);
    }
}

/// <summary>
/// Computation Round Handler
/// </summary>
/// <param name="userData"></param>
void GPUComputation::computationResultHandler(cudaStream_t stream, cudaError_t status, void *userData) {
    // synchronize on stream
    ComputationStateData *computation = static_cast<ComputationStateData *>(userData);

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

    if (computation->norm < CONVERGENCE || last) {
        /// update offset

        auto desidreState = static_cast<ComputationStateData *>(nullptr);
        if (result.compare_exchange_strong(desidreState, computation)) {
            condition.notify_one();
        }
    }

    // synchronize with stream next computation
}

void ConstraintGetFullNorm(size_t coffSize, size_t size, double *b, double *result, cudaStream_t stream) {
    linear_system_method_cuBlas_vectorNorm(static_cast<int>(coffSize), (b + size), result, stream);
}



void GPUComputation::checkStreamNoError() {
    if (settings::get()->DEBUG_CHECK_ARG) {
        checkCudaStatus(cudaStreamSynchronize(stream));
        checkCudaStatus(cudaPeekAtLastError());
    }
}

void GPUComputation::solveSystem(solver::SolverStat *stat, cudaError_t *error) {
    ///
    /// Concept -> Consideration -> ::"ingest stream and observe" fetch first converged
    ///
    ///   -- zalaczamy zadanie FI(q) = 0 norm -> wpiszemy do ExecutionContext
    ///   variable
    ///
    ///   --- data-dest lineage memcpy(deviceToDevice)
    ///

    // register C-function reference delegate
    _registerComputation = this;

    //- fill in A , b, SV

    size_t N = dimension;

    /// !!!  max(max(points.size(), geometrics.size()), constraints.size());

    /// default kernel settings
    const unsigned int ST_DIM_GRID = settings::get()->GRID_SIZE;
    const unsigned int ST_DIM_BLOCK = settings::get()->BLOCK_SIZE;

    solverWatch.setStartTick();

    /// prepare local offset context
    result.store(NULL, std::memory_order_seq_cst);

    checkCudaStatus(cudaStreamSynchronize(stream));

    evaluationWatch.setStopTick();

    /// [ Alternative-Option ] cudaHostRegister ( void* ptr, size_t size, unsigned
    /// int  flags ) :: cudaHostRegisterMapped:

    /// ===============================================
    /// Aync Flow - wszystkie bloki za juz zaincjalizowane
    ///           - adressy blokow sa widoczne on the async-computationHandler-site [publikacja
    ///           adrressow/ arg capturing ]
    ///

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
            utility::memsetAsync(dev_SV[0] + size, 0, coffSize, stream);
        } else {
            checkCudaStatus(
                cudaMemcpyAsync(dev_SV[itr], dev_SV[itr - 1], N * sizeof(double), cudaMemcpyDeviceToDevice, stream));
        }

        ///  Host Context - with references to device

        ComputationStateData **ev = cc->ev;
        ComputationStateData **dev_ev = cc->dev_ev;

        // computation data;
        ev[itr]->cID = itr;
        ev[itr]->SV = dev_SV[itr]; /// data vector lineage
        ev[itr]->A = dev_A;
        ev[itr]->b = dev_b;
        ev[itr]->dx = dev_dx;
        ev[itr]->dev_norm = cc->get_dev_norm(itr);

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

        utility::memcpyAsync(&dev_ev[itr], ev[itr], 1, stream);

        // #observation
        cc->recordComputeStart(itr);

        cc->recordPrepStart(itr);

        /// # KERNEL_PRE

        // if (itr > 0) {
        /// zerujemy macierz A      !!!!! second buffer
        utility::memsetAsync(dev_A, 0, N * N, stream); // --- ze wzgledu na addytywnosc
                                                       //}

        /// macierz `A
        /// Cooficients Stiffnes Matrix
        ComputeStiffnessMatrix<<<ST_DIM_GRID, ST_DIM_BLOCK, Ns, stream>>>(dev_ev[itr], geometrics.size());
        checkStreamNoError();

        /// [ cuda / error ] 701 : cuda API failed 701,
        /// cudaErrorLaunchOutOfResources  = too many resources requested for launch
        //
        if (settings::get()->SOLVER_INC_HESSIAN) {
            EvaluateConstraintHessian<<<ST_DIM_GRID, ST_DIM_BLOCK, Ns, stream>>>(dev_ev[itr], constraints.size());
        }
        checkStreamNoError();

        /*
            Lower Tensor Slice

            --- (Wq); /// Jq = d(Fi)/dq --- write without intermediary matrix
        */

        size_t dimBlock = constraints.size();
        EvaluateConstraintJacobian<<<ST_DIM_GRID, dimBlock, Ns, stream>>>(dev_ev[itr], constraints.size());
        checkStreamNoError();
        /*
            Transposed Jacobian - Uperr Tensor Slice

            --- (Wq)'; /// Jq' = (d(Fi)/dq)' --- transposed - write without
           intermediary matrix
        */
        EvaluateConstraintTRJacobian<<<ST_DIM_GRID, dimBlock, Ns, stream>>>(dev_ev[itr], constraints.size());
        checkStreamNoError();

        /// Tworzymy Macierz dest SV dest `b

        /// [ SV ]  - right hand site

        /// Fr /// Sily  - F(q) --  !!!!!!!
        EvaluateForceIntensity<<<ST_DIM_GRID, ST_DIM_BLOCK, Ns, stream>>>(dev_ev[itr], geometrics.size());
        checkStreamNoError();

        /// Fi / Wiezy  - Fi(q)
        EvaluateConstraintValue<<<ST_DIM_GRID, ST_DIM_BLOCK, Ns, stream>>>(dev_ev[itr], constraints.size());
        checkStreamNoError();

        if (settings::get()->DEBUG_TENSOR_A) {
            stdoutTensorData<<<GRID_DBG, 1, Ns, stream>>>(dev_ev[itr], N, N);
            checkCudaStatus(cudaStreamSynchronize(stream));
        }

        if (settings::get()->DEBUG_TENSOR_B) {
            stdoutRightHandSide<<<GRID_DBG, 1, Ns, stream>>>(dev_ev[itr], N);
            checkCudaStatus(cudaStreamSynchronize(stream));
        }

        cc->recordPrepStop(itr);

        /// ======== DENSE - CuSolver LINER SYSTEM equation CuSolver    === START

        cc->recordSolverStart(itr);

        linear_system_method_cuSolver(dev_A, dev_b, N, stream);

        cc->recordSolverStop(itr);

        /// ======== LINER SYSTEM equation CuSolver    === STOP

        /// uaktualniamy punkty [ SV ] = [ SV ] + [ delta ]
        StateVectorAddDifference<<<DIM_GRID, DIM_BLOCK, Ns, stream>>>(dev_SV[itr], dev_b, N);
        checkStreamNoError();
        // print actual state vector single kernel

        if (settings::get()->DEBUG_TENSOR_SV) {
            stdoutStateVector<<<GRID_DBG, 1, Ns, stream>>>(dev_ev[itr], N);
            checkCudaStatus(cudaStreamSynchronize(stream));
        }
        checkStreamNoError();

        ///  uzupelnic #Question: Vector B = Fi(q) = 0 przeliczamy jeszce raz !!!
        // - not used !!!
        utility::memcpyFromDevice(ev[itr], dev_ev[itr], 1, stream);

        double *host_norm = &ev[itr]->norm;

        ConstraintGetFullNorm(coffSize, size, dev_b, host_norm, stream);
        checkStreamNoError();
        /// ============================================================
        ///   copy DeviceComputation to HostComputation -- addCallback
        /// ============================================================

        void *userData = static_cast<void *>(ev[itr]); // local_Computation

        checkCudaStatus(cudaStreamAddCallback(stream, GPUComputation::computationResultHandlerDelegate, userData, 0));

        cc->recordComputeStop(itr);

        itr++;
    }

    /// all computation tiles submited
    std::unique_lock<std::mutex> ulock(mutex);

    /// atomic read
    if (result.load(std::memory_order_seq_cst) == nullptr) {
        /// spurious wakeup
        condition.wait(ulock, [&] { return result.load(std::memory_order_seq_cst) != nullptr; });
    }

    ComputationState *computation;
    computation = (ComputationState *)result.load(std::memory_order_seq_cst);
    solverWatch.setStopTick();

    // condition met
    // STATE VECTOR offset
    // oczekuje strumienia ale nie kopiujemy danych

    checkCudaStatus(cudaStreamSynchronize(stream));

    // unregister C-function reference delegate
    _registerComputation = nullptr;

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

    utility::memcpyFromDevice(points, d_points, stream);

    double SOLVER_EPSILON = (settings::get()->CU_SOLVER_EPSILON);

    stat->startTime = solverWatch.getStartTick();
    stat->stopTime = solverWatch.getStopTick();
    stat->timeDelta = solverWatch.delta();

    stat->size = size;
    stat->coefficientArity = coffSize;
    stat->dimension = dimension;

    stat->accEvaluationTime = evaluationWatch.delta(); /// !! nasz wewnetrzny allocator pamieci !
    stat->accSolverTime = solverWatch.delta();

    stat->convergence = computation->norm < SOLVER_EPSILON;
    stat->error = computation->norm;
    stat->constraintDelta = computation->norm;
    stat->iterations = computation->cID;

    /// Evaluation data for  device  - CONST DATE for in process execution

    checkCudaStatus(cudaStreamSynchronize(stream));

    *error = cudaGetLastError();
}

double GPUComputation::getPointPXCoordinate(int id) {
    int offset = pointOffset[id];
    double px = points[offset].x;
    return px;
}

double GPUComputation::getPointPYCoordinate(int id) {
    int offset = pointOffset[id];
    double py = points[offset].y;
    return py;
}

void GPUComputation::fillPointCoordinateVector(double *stateVector) {
    for (size_t i = 0, size = points.size(); i < size; i++) {
        graph::Point &p = points[i];
        stateVector[2 * i] = p.x;
        stateVector[2 * i + 1] = p.y;
    }
}

int GPUComputation::updateConstraintState(int constraintId[], double vecX[], double vecY[], int size) {

    evaluationWatch.setStartTick();

    for (int itr = 0; itr < size; itr++) {
        int cId = constraintId[itr];
        int offset = constraintOffset[cId];
        graph::Constraint &constraint = constraints[offset];

        if (constraint.constraintTypeId != CONSTRAINT_TYPE_ID_FIX_POINT) {
            printf("[error] constraint type only supported is ConstraintFixPoint ! \n");
            return 1;
        }
        constraint.vecX = vecX[itr];
        constraint.vecY = vecY[itr];
    }

    if (d_constraints != nullptr) {
        utility::memcpyAsync(&d_constraints, constraints, stream);
    }
    return 0;
}

int GPUComputation::updateParametersValues(int parameterId[], double value[], int size) {
    for (int itr = 0; itr < size; itr++) {
        int pId = parameterId[itr];
        int offset = parameterOffset[pId];
        graph::Parameter &parameter = parameters[offset];
        parameter.value = value[itr];
    }

    if (d_parameters != nullptr) {
        utility::memcpyAsync(&d_parameters, parameters, stream);
    }
    return 0;
}

void GPUComputation::updatePointCoordinateVector(double stateVector[]) {
    for (size_t i = 0, size = points.size(); i < size; i++) {
        graph::Point &p = points[i];
        p.x = stateVector[2 * i];
        p.y = stateVector[2 * i + 1];
    }

    if (d_points != nullptr) {
        utility::memcpyAsync(&d_points, points, stream);
    }
}

//////////////////===================================

GPUComputation::~GPUComputation() {

    /// at least one solver computation

    for (int itr = 0; itr < CMAX; itr++) {
        utility::freeAsync(dev_SV[itr], stream);
    }

    utility::freeAsync(dev_A, stream);
    utility::freeAsync(dev_b, stream);
    utility::freeAsync(dev_dx, stream);

    utility::freeAsync(d_parameterOffset, stream);
    utility::freeAsync(d_accGeometricSize, stream);
    utility::freeAsync(d_accConstraintSize, stream);
    utility::freeAsync(d_constraintOffset, stream);
    utility::freeAsync(d_pointOffset, stream);

    utility::freeAsync(d_points, stream);
    utility::freeAsync(d_geometrics, stream);
    utility::freeAsync(d_constraints, stream);
    utility::freeAsync(d_parameters, stream);
}

} // namespace solver