#include "gpu_computation.h"

#include "cuda_runtime_api.h"

#include <functional>
#include <numeric>
#include <typeinfo>

#include "solver_kernel.cuh"
#include "utility.cuh"

namespace solver {

/// ====================

GPUComputation::GPUComputation(long computationId, cudaStream_t stream, std::shared_ptr<GPULinearSystem> linearSystem,
                               std::shared_ptr<GPUComputationContext> cc, std::vector<graph::Point> &&points,
                               std::vector<graph::Geometric> &&geometrics, std::vector<graph::Constraint> &&constraints,
                               std::vector<graph::Parameter> &&parameters)
    : computationId(computationId), _linearSystem(linearSystem), _cc(cc), _stream(stream), _points(std::move(points)),
      _geometrics(std::move(geometrics)), _constraints(std::move(constraints)), _parameters(std::move(parameters)) {
    if (_points.size() == 0) {
        printf("empty solution space, add some geometric types\n");
        error = cudaSuccess;
        return;
    }

    // model aggreate for const immutable data
    if (_points.size() == 0) {
        printf("[solver] - empty evaluation space model\n");
        error = cudaSuccess;
        return;
    }

    if (_constraints.size() == 0) {
        printf("[solver] - no constraint configuration applied onto model\n");
        error = cudaSuccess;
        return;
    }

    evaluationWatch.setStartTick();

    dev_SV = std::vector<double *>(CMAX, 0);

    /// setup all dependent structure for device , accSize or accOffset
    preInitializeData();

    /// prepera allocation and transfer structures onto device
    memcpyComputationToDevice();

    printf("[solver] model stage consistent !\n");
}

//////////////////===================================

void GPUComputation::preInitializeData() {

    /// mapping from point Id => point dest offset
    pointOffset = utility::stateOffset(_points, [](auto point) { return point->id; });

    geometricOffset = utility::stateOffset(_geometrics, [](auto geometric) { return geometric->id; });

    constraintOffset = utility::stateOffset(_constraints, [](auto constraint) { return constraint->id; });

    /// this is mapping from Id => parameter dest offset
    parameterOffset = utility::stateOffset(_parameters, [](auto parameter) { return parameter->id; });

    /// accumalted position of geometric block
    accGeometricSize = utility::accumalatedValue(_geometrics, graph::geometricSetSize);

    /// accumulated position of constrain block
    accConstraintSize = utility::accumalatedValue(_constraints, graph::constraintSize);

    /// `A` tensor internal structure dimensions
    size = std::accumulate(_geometrics.begin(), _geometrics.end(), 0,
                           [](auto acc, auto const &geometric) { return acc + graph::geometricSetSize(geometric); });

    coffSize = std::accumulate(_constraints.begin(), _constraints.end(), 0, [](auto acc, auto const &constraint) {
        return acc + graph::constraintSize(constraint);
    });

    dimension = size + coffSize;
}

void GPUComputation::memcpyComputationToDevice() {

    /// ============================================================
    ///         Host Computation with references to Device
    /// ============================================================

    /// const data in computation
    utility::mallocAsync(&d_points, _points.size(), _stream);
    utility::mallocAsync(&d_geometrics, _geometrics.size(), _stream);
    utility::mallocAsync(&d_constraints, _constraints.size(), _stream);
    utility::mallocAsync(&d_parameters, _parameters.size(), _stream);

    utility::mallocAsync(&d_pointOffset, pointOffset.size(), _stream);
    utility::mallocAsync(&d_geometricOffset, geometricOffset.size(), _stream);
    utility::mallocAsync(&d_constraintOffset, constraintOffset.size(), _stream);
    utility::mallocAsync(&d_parameterOffset, parameterOffset.size(), _stream);
    utility::mallocAsync(&d_accGeometricSize, accGeometricSize.size(), _stream);
    utility::mallocAsync(&d_accConstraintSize, accConstraintSize.size(), _stream);

    // immutables -
    utility::memcpyAsync(&d_points, _points, _stream);
    utility::memcpyAsync(&d_geometrics, _geometrics, _stream);
    utility::memcpyAsync(&d_constraints, _constraints, _stream);
    utility::memcpyAsync(&d_parameters, _parameters, _stream);

    // immutables
    utility::memcpyAsync(&d_pointOffset, pointOffset.data(), pointOffset.size(), _stream);
    utility::memcpyAsync(&d_geometricOffset, geometricOffset.data(), geometricOffset.size(), _stream);
    utility::memcpyAsync(&d_constraintOffset, constraintOffset.data(), constraintOffset.size(), _stream);

    utility::memcpyAsync(&d_accGeometricSize, accGeometricSize.data(), accGeometricSize.size(), _stream);
    utility::memcpyAsync(&d_accConstraintSize, accConstraintSize.data(), accConstraintSize.size(), _stream);
    if (!_parameters.empty()) {
        utility::memcpyAsync(&d_parameterOffset, parameterOffset.data(), parameterOffset.size(), _stream);
    }

    size_t N = dimension;
    ///
    ///  [ GPU ] tensors `A` `x` `dx` `b`  ------------------------
    ///
    utility::mallocAsync(&dev_A, N * N, _stream);
    utility::mallocAsync(&dev_b, N, _stream);
    utility::mallocAsync(&dev_dx, N, _stream);

    for (int itr = 0; itr < CMAX; itr++) {
        /// each computation data with its own StateVector
        utility::mallocAsync(&dev_SV[itr], N, _stream);
    }
}

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

void GPUComputation::checkStreamNoError() {
    if (settings::get()->DEBUG_CHECK_ARG) {
        checkCudaStatus(cudaStreamSynchronize(_stream));
        checkCudaStatus(cudaPeekAtLastError());
    }
}

GPUComputation *GPUComputation::_registerComputation = nullptr;

void GPUComputation::solveSystem(solver::SolverStat *stat, cudaError_t *error) {
    ///
    /// Concept -> Consideration -> ::"ingest stream and observe" fetch first converged
    ///
    ///   -- zalaczamy zadanie FI(q) = 0 norm -> wpiszemy do ExecutionContext
    ///   variable
    ///
    ///   --- data-dest lineage memcpy(deviceToDevice)
    ///

    // register C-function reference delegate - registerForDelegation(this) / unregisterFromDelegation(this)
    GPUComputation::_registerComputation = this;

    //- fill in A , b, SV

    size_t N = dimension;

    /// !!!  max(max(points.size(), geometrics.size()), constraints.size());

    /// default kernel settings
    unsigned int ST_DIM_GRID = settings::get()->GRID_SIZE;
    unsigned int ST_DIM_BLOCK = settings::get()->BLOCK_SIZE;

    solverWatch.setStartTick();

    /// prepare local offset context
    result.store(NULL, std::memory_order_seq_cst);

    checkCudaStatus(cudaStreamSynchronize(_stream));

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
        
        _cc->recordComputeStart(itr);

        if (itr == 0) {
            /// SV - State Vector

            CopyIntoStateVector<<<ST_DIM_GRID, ST_DIM_BLOCK, Ns, _stream>>>(dev_SV[0], d_points, size);

            /// SV -> setup Lagrange multipliers  -
            utility::memsetAsync(dev_SV[0] + size, 0, coffSize, _stream);
        } else {
            checkCudaStatus(
                cudaMemcpyAsync(dev_SV[itr], dev_SV[itr - 1], N * sizeof(double), cudaMemcpyDeviceToDevice, _stream));
        }

        ///  Host Context - with references to device

        std::vector<ComputationStateData *> &ev = _cc->ev;
        std::vector<ComputationStateData *> &dev_ev = _cc->dev_ev;

        // computation data;
        ev[itr]->cID = itr;
        ev[itr]->SV = dev_SV[itr]; /// data vector lineage
        ev[itr]->A = dev_A;
        ev[itr]->b = dev_b;
        ev[itr]->dx = dev_dx;
        ev[itr]->dev_norm = _cc->get_dev_norm(itr);

        // geometric structure
        ev[itr]->size = size;
        ev[itr]->coffSize = coffSize;
        ev[itr]->dimension = dimension = N;

        ev[itr]->points = NVector<graph::Point>(d_points, _points.size());
        ev[itr]->geometrics = NVector<graph::Geometric>(d_geometrics, _geometrics.size());
        ev[itr]->constraints = NVector<graph::Constraint>(d_constraints,_constraints.size());
        ev[itr]->parameters = NVector<graph::Parameter>(d_parameters, _parameters.size());

        ev[itr]->pointOffset = d_pointOffset;
        ev[itr]->geometricOffset = d_geometricOffset;
        ev[itr]->constraintOffset = d_constraintOffset;
        ev[itr]->parameterOffset = d_parameterOffset;
        ev[itr]->accGeometricSize = d_accGeometricSize;
        ev[itr]->accConstraintSize = d_accConstraintSize;

        ///
        /// [ GPU ] computation context mapped onto devive object
        ///
        /// tu chce snapshot transfer

        utility::memcpyAsync(&dev_ev[itr], ev[itr], 1, _stream);       
        checkStreamNoError();


        // #observation
        

        _cc->recordPrepStart(itr);

        /// # KERNEL_PRE

        // if (itr > 0) {
        
        /// zerujemy macierz A      !!!!! second buffer
        utility::memsetAsync(dev_A, 0, N * N, _stream); // --- ze wzgledu na addytywnosc


       
        if (settings::get()->KERNEL_PRE) {


            /*
            *  Matrix A, b vector production kernel
            */
            /// computation threads requirments
            unsigned int TRDS = 3 * _geometrics.size() + 3 * _constraints.size();
            unsigned int BLOCK_DIM = (TRDS + ST_DIM_BLOCK - 1) / ST_DIM_BLOCK;
            BuildComputationMatrix<<<BLOCK_DIM, ST_DIM_BLOCK, Ns, _stream>>>(dev_ev[itr], _geometrics.size(),
                                                                               _constraints.size());

        } else {  

            
            /// BEBUG - KERNEL

            /// macierz `A
            /// Cooficients Stiffnes Matrix
            ComputeStiffnessMatrix<<<ST_DIM_GRID, ST_DIM_BLOCK, Ns, _stream>>>(dev_ev[itr], _geometrics.size());
            checkStreamNoError();

            /// [ cuda / error ] 701 : cuda API failed 701,
            /// cudaErrorLaunchOutOfResources  = too many resources requested for launch
            //
            if (settings::get()->SOLVER_INC_HESSIAN) {
                EvaluateConstraintHessian<<<ST_DIM_GRID, ST_DIM_BLOCK, Ns, _stream>>>(dev_ev[itr], _constraints.size());
            }
            checkStreamNoError();

            /*
                Lower Tensor Slice

                --- (Wq); /// Jq = d(Fi)/dq --- write without intermediary matrix
            */

            EvaluateConstraintJacobian<<<ST_DIM_GRID, ST_DIM_BLOCK, Ns, _stream>>>(dev_ev[itr], _constraints.size());
            checkStreamNoError();
            /*
                Transposed Jacobian - Uperr Tensor Slice

                --- (Wq)'; /// Jq' = (d(Fi)/dq)' --- transposed - write without
               intermediary matrix
            */

            EvaluateConstraintTRJacobian<<<ST_DIM_GRID, ST_DIM_BLOCK, Ns, _stream>>>(dev_ev[itr], _constraints.size());
            checkStreamNoError();

            /// Tworzymy Macierz dest SV dest `b

            /// [ SV ]  - right hand site

            /// Fr /// Sily  - F(q) --  !!!!!!!
            EvaluateForceIntensity<<<ST_DIM_GRID, ST_DIM_BLOCK, Ns, _stream>>>(dev_ev[itr], _geometrics.size());
            checkStreamNoError();

            /// Fi / Wiezy  - Fi(q)
            EvaluateConstraintValue<<<ST_DIM_GRID, ST_DIM_BLOCK, Ns, _stream>>>(dev_ev[itr], _constraints.size());
            checkStreamNoError();
        }

/// 
        _cc->recordPrepStop(itr);


        if (settings::get()->DEBUG_TENSOR_A) {
            stdoutTensorData<<<GRID_DBG, 1, Ns, _stream>>>(dev_ev[itr], N, N);
            checkCudaStatus(cudaStreamSynchronize(_stream));
        }

        if (settings::get()->DEBUG_TENSOR_B) {
            stdoutRightHandSide<<<GRID_DBG, 1, Ns, _stream>>>(dev_ev[itr], N);
            checkCudaStatus(cudaStreamSynchronize(_stream));
        }


        _cc->recordSolverStart(itr);

        ///  uzupelnic #Question: Vector B = Fi(q) = 0 przeliczamy jeszce raz !!!
        /// - not used !!! !!
        ///
        ///
        /// utility::memcpyFromDevice(ev[itr], dev_ev[itr], 1, _stream);

        double *host_norm = &ev[itr]->norm;

        //
        ///  ConstraintGetFullNorm
        //
        _linearSystem->vectorNorm(static_cast<int>(coffSize), (dev_b + size), host_norm);

        checkStreamNoError();
                
        /// ======== DENSE - CuSolver LINER SYSTEM equation CuSolver    === START

        _linearSystem->solveLinearEquation(dev_A, dev_b, N);

        /// ======== LINER SYSTEM equation CuSolver    === STOP
        _cc->recordSolverStop(itr);

        /// uaktualniamy punkty [ SV ] = [ SV ] + [ delta ]
        StateVectorAddDifference<<<DIM_GRID, ST_DIM_BLOCK, Ns, _stream>>>(dev_SV[itr], dev_b, N);
        checkStreamNoError();
        // print actual state vector single kernel

        if (settings::get()->DEBUG_TENSOR_SV) {
            stdoutStateVector<<<GRID_DBG, 1, Ns, _stream>>>(dev_ev[itr], N);
            checkCudaStatus(cudaStreamSynchronize(_stream));
        }
        checkStreamNoError();


        /// ============================================================
        ///   copy DeviceComputation to HostComputation -- addCallback
        /// ============================================================

        void *userData = static_cast<void *>(ev[itr]); // local_Computation

        checkCudaStatus(cudaStreamAddCallback(_stream, GPUComputation::computationResultHandlerDelegate, userData, 0));

        _cc->recordComputeStop(itr);

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

    checkCudaStatus(cudaStreamSynchronize(_stream));

    // unregister C-function reference delegate
    GPUComputation::_registerComputation = nullptr;

    if (settings::get()->DEBUG) {

        reportThisResult(computation);
    }

    ///
    /// UWAGA !!! 3 dni poszukiwania bledu -  "smigamy po allokacjach"
    /// 
    ///     uVector operator[]() { IF DEBUG bound check for  illegal access }
    /// 
    CopyFromStateVector<<<DIM_GRID, ST_DIM_BLOCK, Ns, _stream>>>(d_points, computation->SV, _points.size());

    utility::memcpyFromDevice(_points, d_points, _stream);

    double SOLVER_EPSILON = (settings::get()->CU_SOLVER_EPSILON);

    int iter = computation->cID;

    stat->startTime = solverWatch.getStartTick();
    stat->stopTime = solverWatch.getStopTick();
    stat->timeDelta = _cc->getAccComputeTime(iter); /// solverWatch.delta() + evaluationWatch.delta();

    stat->size = size;
    stat->coefficientArity = coffSize;
    stat->dimension = dimension;

    stat->accEvaluationTime  = _cc->getAccPrepTime(iter); // evaluationWatch.delta(); /// !! nasz wewnetrzny allocator pamieci !    
    stat->accSolverTime = _cc->getAccSolverTime(iter);  // solverWatch.delta();

    stat->convergence = computation->norm < SOLVER_EPSILON;
    stat->error = computation->norm;
    stat->constraintDelta = computation->norm;
    stat->iterations = computation->cID;

    /// Evaluation data for  device  - CONST DATE for in process execution

    checkCudaStatus(cudaStreamSynchronize(_stream));

    solverWatch.reset();
    evaluationWatch.reset();


    *error = cudaGetLastError();
}

void GPUComputation::reportThisResult(ComputationStateData *computation) {
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

double GPUComputation::getPointPXCoordinate(int id) {
    int offset = pointOffset[id];
    double px = _points[offset].x;
    return px;
}

double GPUComputation::getPointPYCoordinate(int id) {
    int offset = pointOffset[id];
    double py = _points[offset].y;
    return py;
}

void GPUComputation::fillPointCoordinateVector(double *stateVector) {
    for (size_t i = 0, size = _points.size(); i < size; i++) {
        graph::Point &p = _points[i];
        stateVector[2 * i] = p.x;
        stateVector[2 * i + 1] = p.y;
    }
}

int GPUComputation::updateConstraintState(int constraintId[], double vecX[], double vecY[], int size) {

    evaluationWatch.setStartTick();

    for (int itr = 0; itr < size; itr++) {
        int cId = constraintId[itr];
        int offset = constraintOffset[cId];
        graph::Constraint &constraint = _constraints[offset];

        if (constraint.constraintTypeId != CONSTRAINT_TYPE_ID_FIX_POINT) {
            printf("[error] constraint type only supported is ConstraintFixPoint ! \n");
            return 1;
        }
        constraint.vecX = vecX[itr];
        constraint.vecY = vecY[itr];
    }

    if (d_constraints != nullptr) {
        utility::memcpyAsync(&d_constraints, _constraints, _stream);
    }
    return 0;
}

int GPUComputation::updateParametersValues(int parameterId[], double value[], int size) {
    for (int itr = 0; itr < size; itr++) {
        int pId = parameterId[itr];
        int offset = parameterOffset[pId];
        graph::Parameter &parameter = _parameters[offset];
        parameter.value = value[itr];
    }

    if (d_parameters != nullptr) {
        utility::memcpyAsync(&d_parameters, _parameters, _stream);
    }
    return 0;
}

void GPUComputation::updatePointCoordinateVector(double stateVector[]) {
    for (size_t i = 0, size = _points.size(); i < size; i++) {
        graph::Point &p = _points[i];
        p.x = stateVector[2 * i];
        p.y = stateVector[2 * i + 1];
    }

    if (d_points != nullptr) {
        utility::memcpyAsync(&d_points, _points, _stream);
    }
}

//////////////////===================================

GPUComputation::~GPUComputation() {

    /// at least one solver computation

    for (int itr = 0; itr < CMAX; itr++) {
        utility::freeAsync(dev_SV[itr], _stream);
    }

    utility::freeAsync(dev_A, _stream);
    utility::freeAsync(dev_b, _stream);
    utility::freeAsync(dev_dx, _stream);

    utility::freeAsync(d_parameterOffset, _stream);
    utility::freeAsync(d_accGeometricSize, _stream);
    utility::freeAsync(d_accConstraintSize, _stream);
    utility::freeAsync(d_constraintOffset, _stream);
    utility::freeAsync(d_geometricOffset, _stream);
    utility::freeAsync(d_pointOffset, _stream);

    utility::freeAsync(d_points, _stream);
    utility::freeAsync(d_geometrics, _stream);
    utility::freeAsync(d_constraints, _stream);
    utility::freeAsync(d_parameters, _stream);
}

} // namespace solver