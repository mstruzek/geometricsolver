#include "com_mstruzek_jni_JNISolverGate.h"

#include <algorithm>
#include <chrono>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <vector>

#include "model.h"

/// GPU common variables
static int deviceId;
static cudaError_t error_t;

/// points register
static std::vector<Point> points; /// poLocations id-> point_offset

/// geometricc register
static std::vector<Geometric> geometrics; /// ==> Macierz A, accumulative offset for each primitive

/// constraints register
static std::vector<Constraint> constraints; /// ===> Wiezy , accumulative offset for each constraint

/// parameters register
static std::vector<Parameter> parameters; /// paramLocation id-> param_offset

/// Point Locations in computation matrix [id] -> point offset
static int *poLocations;

/// Parameter locations [id] -> parameter offset
static int *paramLocations;

static SolverStat stat;

/**
 * @brief  setup all matricies for computation and prepare kernel stream  intertwined with cusolver
 *
 */
void solveSystemOnGPU(int *error);

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    initDriver
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_initDriver(JNIEnv *env, jclass clazz) {
        int count;

        error_t = cudaGetDeviceCount(&count);
        if (error_t != cudaSuccess) {
                printf("driver  error %d  = %s \n", static_cast<int>(error_t), cudaGetErrorString(error_t));
                return JNI_ERROR;
        }

        deviceId = 0;

        error_t = cudaSetDevice(deviceId);
        if (error_t != cudaSuccess) {
                printf("driver  error %d  = %s \n", static_cast<int>(error_t), cudaGetErrorString(error_t));
                return JNI_ERROR;
        }
        return JNI_SUCCESS;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    getLastError
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_mstruzek_jni_JNISolverGate_getLastError(JNIEnv *env, jclass clazz) {
        /// cuda error

        const char *msg = cudaGetErrorString(error_t);

        return env->NewStringUTF(msg);

        /// cusolver error
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    closeDriver
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_closeDriver(JNIEnv *env, jclass clazz) {
        error_t = cudaDeviceReset();
        if (error_t != cudaSuccess) {
                printf("driver  error %d  = %s \n", static_cast<int>(error_t), cudaGetErrorString(error_t));
                return JNI_ERROR;
        }
        return JNI_SUCCESS;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    resetDatabase
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_resetComputationData(JNIEnv *env, jclass clazz) {
        std::remove_if(points.begin(), points.end(), [](auto _) { return true; });
        std::remove_if(geometrics.begin(), geometrics.end(), [](auto _) { return true; });
        std::remove_if(constraints.begin(), constraints.end(), [](auto _) { return true; });
        std::remove_if(parameters.begin(), parameters.end(), [](auto _) { return true; });

        error_t = cudaFreeHost(poLocations);
        if (error_t != cudaSuccess) {
                printf("free mem error %d  = %s \n", static_cast<int>(error_t), cudaGetErrorString(error_t));
                return JNI_ERROR;
        }

        error_t = cudaFreeHost(paramLocations);
        if (error_t != cudaSuccess) {
                printf("free mem error %d  = %s \n", static_cast<int>(error_t), cudaGetErrorString(error_t));
                return JNI_ERROR;
        }

        poLocations = NULL;
        paramLocations = NULL;

        return JNI_SUCCESS;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    resetComputationContext
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_resetComputationContext(JNIEnv *env, jclass clazz) {
        /// workspace - zwolnic pamiec i wyzerowac wskazniki , \\cusolver
        return JNI_SUCCESS;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    initComputationContext
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_initComputationContext(JNIEnv *env, jclass clazz) {

        /// po zarejestrowaniu calego modelu w odpowiadajacych rejestrach , zainicjalizowac pomocnicze macierze
        /// przygotowac zmienne dla [cusolvera]

        /// przeliczenie pozycji absolutnej punktu na macierzy wyjsciowej

        return JNI_SUCCESS;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    solveSystem
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_solveSystem(JNIEnv *env, jclass clazz) {

        int error;

        error = 0;

        solveSystemOnGPU(&error);

        if (error != 0) {
                return JNI_ERROR;
        }

        return JNI_SUCCESS;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    getSolverStatistics
 * Signature: ()Lcom/mstruzek/msketch/solver/SolverStat;
 */
JNIEXPORT jobject JNICALL Java_com_mstruzek_jni_JNISolverGate_getSolverStatistics(JNIEnv *env, jclass) {
        jclass objClazz = env->FindClass("com/mstruzek/msketch/solver/SolverStat");
        if (objClazz == NULL) {
                printf("SolverStat is not visible in class loader\n");
                env->ThrowNew(env->FindClass("java/lang/RuntimeException"), "SolverStat is not visible in class loader");
                return 0;
        }

        jmethodID defaultCtor = env->GetMethodID(objClazz, "<init>", "()V");
        if (defaultCtor == NULL) {
                printf("constructor not visible\n");
                env->ThrowNew(env->FindClass("java/lang/RuntimeException"), "SolverStat constructor not visible");
                return 0;
        }

        jobject objStat = env->NewObject(objClazz, defaultCtor);
        if (objStat == NULL) {
                printf("object statt <init> error\n");
                env->ThrowNew(env->FindClass("java/lang/RuntimeException"), "construction instance error");
                return 0;
        }
        env->SetLongField(objStat, env->GetFieldID(objClazz, "startTime", "J"), stat.startTime);
        env->SetLongField(objStat, env->GetFieldID(objClazz, "stopTime", "J"), stat.stopTime);
        env->SetLongField(objStat, env->GetFieldID(objClazz, "timeDelta", "J"), stat.timeDelta);
        env->SetIntField(objStat, env->GetFieldID(objClazz, "size", "I"), stat.size);
        env->SetIntField(objStat, env->GetFieldID(objClazz, "coefficientArity", "I"), stat.coefficientArity);
        env->SetIntField(objStat, env->GetFieldID(objClazz, "dimension", "I"), stat.dimension);
        env->SetLongField(objStat, env->GetFieldID(objClazz, "accEvaluationTime", "J"), stat.accEvaluationTime);
        env->SetLongField(objStat, env->GetFieldID(objClazz, "accSolverTime", "J"), stat.accSolverTime);
        env->SetBooleanField(objStat, env->GetFieldID(objClazz, "convergence", "Z"), stat.convergence);
        env->SetDoubleField(objStat, env->GetFieldID(objClazz, "error", "D"), stat.error);
        env->SetDoubleField(objStat, env->GetFieldID(objClazz, "constraintDelta", "D"), stat.constraintDelta);
        env->SetIntField(objStat, env->GetFieldID(objClazz, "iterations", "I"), stat.iterations);
        return objStat;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    registerPointType
 * Signature: (IDD)I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_registerPointType(JNIEnv *env, jclass clazz, jint id, jdouble px, jdouble py) {
        points.emplace_back(id, px, py);
        return JNI_SUCCESS;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    registerGeometricType
 * Signature: (IIIIIIII)I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_registerGeometricType(JNIEnv *env, jclass clazz, jint id, jint geometricTypeId, jint p1, jint p2,
                                                                                 jint p3, jint a, jint b, jint c, jint d) {
        geometrics.emplace_back(id, geometricTypeId, p1, p2, p3, a, b, c, d);
        return JNI_SUCCESS;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    registerParameterType
 * Signature: (ID)I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_registerParameterType(JNIEnv *env, jclass clazz, jint id, jdouble value) {
        parameters.emplace_back(id, value);
        return JNI_SUCCESS;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    registerConstraintType
 * Signature: (IIIIIIDD)I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_registerConstraintType(JNIEnv *env, jclass clazz, jint id, jint jconstraintTypeId, jint k, jint l,
                                                                                  jint m, jint n, jint paramId, jdouble vecX, jdouble vecY) {
        constraints.emplace_back(id, jconstraintTypeId, k, l, m, n, paramId, vecX, vecY);
        return JNI_SUCCESS;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    getPointPXCoordinate
 * Signature: (I)D
 */
JNIEXPORT jdouble JNICALL Java_com_mstruzek_jni_JNISolverGate_getPointPXCoordinate(JNIEnv *env, jclass clazz, jint id) {
        int offset = poLocations[id];
        double px = points[offset].px;
        return (jdouble)px;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    getPointPYCoordinate
 * Signature: (I)D
 */
JNIEXPORT jdouble JNICALL Java_com_mstruzek_jni_JNISolverGate_getPointPYCoordinate(JNIEnv *env, jclass clazz, jint id) {
        int offset = poLocations[id];
        double py = points[offset].py;
        return (jdouble)py;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    getPointCoordinateVector
 * Signature: ()[D
 */
JNIEXPORT jdoubleArray JNICALL Java_com_mstruzek_jni_JNISolverGate_getPointCoordinateVector(JNIEnv *env, jclass clazz) {
        /// depraceted or remove !
        return (jdoubleArray)NULL;
}

struct MatrixDouble {};

long TimeNanosecondsNow() { return std::chrono::duration<long, std::nano>(std::chrono::high_resolution_clock::now().time_since_epoch()).count(); }

class StopWatch {
      public:
        StopWatch() { reset(); }

        void setStartTick() { startTick = TimeNanosecondsNow(); }

        void setStopTick() {
                stopTick = TimeNanosecondsNow();
                accTime += (stopTick - startTick);
        }

        long delta() { return stopTick - startTick; }

        void reset() {

                startTick = 0L;
                stopTick = 0L;
                accTime = 0L;
        }

        long startTick;
        long stopTick;
        long accTime;
};

static StopWatch solverWatch;
static StopWatch accEvoWatch;
static StopWatch accLUWatch;

/// CPU#
long AllLagrangeCoffSize() { return 0; }

/// KERNEL#
double ConstraintGetFullNorm() { return 0.0; }

/// CPU#
void PointLocationSetup() {}

/// CPUtoGPU#
void CopyIntoStateVector(void *) {}

/// CPU# and GPU#
void SetupLagrangeMultipliers(void) {}

/// KERNEL#
void GeometricObjectEvaluateForceVector() { /// Sily  - F(q)
        // b.mulitply(-1);
}

// KERNEL#
void ConstraintEvaluateConstraintVector(){
    /// Wiezy  - Fi(q)

    /// b.mulitply(-1);
};

#define MAX_SOLVER_ITERATIONS 20

#define CONVERGENCE_LIMIT 10e-5

/**
 *  /// wydzielic pliku CU
 *
 *  /// przygotowac petle
 *
 *  ///
 *
 */
void solveSystemOnGPU(int *error) {

        int size;      /// wektor stanu
        int coffSize;  /// wspolczynniki Lagrange
        int dimension; /// dimension = size + coffSize

        solverWatch.reset();
        accEvoWatch.reset();
        accLUWatch.reset();

        /// Uklad rownan liniowych  [ A * x = b ] powstały z linerazycji ukladu dynamicznego

/// 1# CPU -> GPU inicjalizacja macierzy

        MatrixDouble A;  /// Macierz głowna ukladu rownan liniowych
        MatrixDouble Fq; /// Macierz sztywnosci ukladu obiektow zawieszonych na sprezynach.
        MatrixDouble Wq; /// d(FI)/dq - Jacobian Wiezow

        /// HESSIAN
        MatrixDouble Hs;

        // Wektor prawych stron [Fr; Fi]'
        MatrixDouble b;

        // skladowe to Fr - oddzialywania pomiedzy obiektami, sily w sprezynach
        MatrixDouble Fr;

        // skladowa to Fiq - wartosci poszczegolnych wiezow
        MatrixDouble Fi;

        double norm1;            /// wartosci bledow na wiezach
        double prevNorm;         /// norma z wczesniejszej iteracji,
        double errorFluctuation; /// fluktuacja bledu

        stat.startTime = TimeNanosecondsNow();

        printf("@#=================== Solver Initialized ===================#@ \n");
        printf("");

        if (points.size() == 0) {
                printf("[warning] - empty model\n");
                return;
        }

        if (constraints.size()) {
                printf("[warning] - no constraint configuration applied \n");
                return;
        }

        /// Tworzymy Macierz "A" - dla tego zadania stala w czasie

        size = points.size() * 2;
        coffSize = AllLagrangeCoffSize();
        dimension = size + coffSize;

        stat.size = size;
        stat.coefficientArity = coffSize;
        stat.dimension = dimension;

        accEvoWatch.setStartTick();

        /// Inicjalizacje bazowych macierzy rzadkich - SparseMatrix

        /// --->
        // instead loop over vector space provide static access to point location in VS.
        // - reference locations for Jacobian and Hessian evaluation !
        PointLocationSetup();

        /// cMalloc() --- przepisac na GRID -->wspolrzedne

        A = MatrixDouble.matrix2D(dimension, dimension, 0.0);

        Fq = MatrixDouble.matrix2D(size, size, 0.0);

        Wq = MatrixDouble.matrix2D(coffSize, size, 0.0);
        Hs = MatrixDouble.matrix2D(size, size, 0.0);

        /// KERNEL_O

        /// ### macierz sztywnosci stala w czasie - jesli bezkosztowe bliskie zero to zawieramy w KELNER_PRE 
          /// ( dla uproszczenia tylko w pierwszy przejsciu -- Thread_0_Grid_0 - synchronization Guard )

        /// inicjalizacja i bezposrednio do A
        ///   w nastepnych przejsciach -> kopiujemy 


        KernelEvaluateStiffnessMatrix<<<>>>(Fq)


            /// cMallloc

            /// Wektor prawych stron b

            b = MatrixDouble.matrix1D(dimension, 0.0);

        // right-hand side vector ~ b

        Fr = b.viewSpan(0, 0, size, 1);
        Fi = b.viewSpan(size, 0, coffSize, 1);

        MatrixDouble dmx = null;

        /// State Vector - zmienne stanu
        MatrixDouble SV = MatrixDouble.matrix1D(dimension, 0.0);

        /// PointUtility.

        CopyIntoStateVector(NULL);      // SV
        SetupLagrangeMultipliers(); // SV

        accEvoWatch.setStopTick();

        norm1 = 0.0;
        prevNorm = 0.0;
        errorFluctuation = 0.0;

        int itr = 0;
        while (itr < MAX_SOLVER_ITERATIONS) {


                accEvoWatch.setStartTick();

/// --- KERNEL_PRE 

                /// zerujemy macierz A

                /// # KERNEL_PRE

                A.reset(0.0); /// --- ze wzgledu na addytywnosc 

                /// Tworzymy Macierz vector b vector `b

                /// # KERNEL_PRE
                GeometricObjectEvaluateForceVector(Fr); /// Sily  - F(q)
                ConstraintEvaluateConstraintVector(Fi); /// Wiezy  - Fi(q)
                // b.setSubMatrix(0,0, (Fr));
                // b.setSubMatrix(size,0, (Fi));
                b.mulitply(-1);

                /// macierz `A

                /// # KERNEL_PRE (__shared__ JACOBIAN)

                ConstraintGetFullJacobian(Wq); /// Jq = d(Fi)/dq --- write without intermediary matrix   `A = `A set value

                
                Hs.reset(0.0); /// ---- niepotrzebnie


                /// # KERNEL_PRE

                ConstraintGetFullHessian(Hs, SV, size); // --- write without intermediate matrix '`A = `A +value

                A.setSubMatrix(0, 0, Fq);  /// procedure SET
                A.plusSubMatrix(0, 0, Hs); /// procedure ADD

                A.setSubMatrix(size, 0, Wq);
                A.setSubMatrix(0, size, Wq.transpose());

                /*
                 *  LU Decomposition  -- Colt Linear Equation Solver
                 *
                 *   rozwiazjemy zadanie [ A ] * [ dx ] = [ b ]
                 */

                /// DENSE MATRIX - pierwsze podejscie !

                DoubleMatrix2D matrix2DA; /// cuSolver
                DoubleMatrix1D matrix1Db; /// cuSolver


/// ---- KERNEL_PRE

                accEvoWatch.setStopTick();

                accLUWatch.setStartTick();

                /// DENSE - CuSolver
                /// LU Solver
                LUDecompositionQuick LU = new LUDecompositionQuick();
                LU.decompose(matrix2DA);
                LU.solve(matrix1Db);

                accLUWatch.setStopTick();

                /// Bind delta-x into database points

                /// --- KERNEL_POST
                dmx = MatrixDouble.matrixDoubleFrom(matrix1Db);

                /// uaktualniamy punkty [ SV ] = [ SV ] + [ delta ]

                /// --- KERNEL_POST
                SV.plus(dmx);


                PointUtilityCopyFromStateVector(SV); /// ??????? - niepotrzebnie , NA_KONCU
               

                /// --- KERNEL_POST( __synchronize__ REDUCTION )

                norm1 = ConstraintGetFullNorm();

                /// cCopySymbol()

                printf(" [ step :: %02d]  duration [ns] = %,12d  norm = %e \n", itr, (accLUWatch.stopTick - accEvoWatch.startTick), norm1);

                /// Gdy po 5-6 przejsciach iteracji, normy wiezow kieruja sie w strone minimum energii, to repozycjonowac prowadzace punkty

                if (norm1 < CONVERGENCE_LIMIT) {
                        stat.error = norm1;
                        printf("fast convergence - norm [ %e ] \n", norm1);
                        printf("constraint error = %e \n", norm1);
                        printf("");
                        break;
                }

                /// liczymy zmiane bledu
                errorFluctuation = norm1 - prevNorm;
                prevNorm = norm1;
                stat.error = norm1;

                if (itr > 1 && errorFluctuation / prevNorm > 0.70) {
                        printf("CHANGES - STOP ITERATION *******");
                        printf(" errorFluctuation          : %d \n", errorFluctuation);
                        printf(" relative error            : %f \n", (errorFluctuation / norm1));
                        solverWatch.setStopTick();
                        stat.constraintDelta = ConstraintGetFullNorm();
                        stat.convergence = false;
                        stat.stopTime = TimeNanosecondsNow();
                        stat.iterations = itr;
                        stat.accSolverTime = accLUWatch.accTime;
                        stat.accEvaluationTime = accEvoWatch.accTime;
                        stat.timeDelta = solverWatch.stopTick - solverWatch.startTick;
                        return;
                }
                itr++;
        }

        solverWatch.setStopTick();
        long solutionDelta = solverWatch.delta();

        printf("# solution delta : %d \n", solutionDelta);    // print execution time
        printf("\n");                                         // print execution time

        stat.constraintDelta = ConstraintGetFullNorm();
        stat.convergence = norm1 < CONVERGENCE_LIMIT;
        stat.stopTime = TimeNanosecondsNow();
        stat.iterations = itr;
        stat.accSolverTime = accLUWatch.accTime;
        stat.accEvaluationTime = accEvoWatch.accTime;
        stat.timeDelta = solverWatch.stopTick - solverWatch.startTick;

        return;
}