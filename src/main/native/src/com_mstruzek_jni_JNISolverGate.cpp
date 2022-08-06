#include "com_mstruzek_jni_JNISolverGate.h"

#include <cuda_runtime_api.h>
#include <stdio.h>
#include <vector>
#include <algorithm>

static int deviceId;
static cudaError_t error_t;


struct Point {
        int id;
        double px;
        double py;
};

struct Geometric {
        int id;
        int geometricTypeId;
        int p1;
        int p2;
        int p3;
        int a;
        int b;
        int c;
        int d;
};

struct Constraint {
        int id;
        int constrajintTypeId;
        int k;
        int l;
        int m;
        int n;
        int paramId;
        double vecX;
        double vecY;
};

struct Parameter {
        int id;
        double value;
};


/// points register
static std::vector<Point> points;             /// poLocations id-> point_offset

/// geometricc register
static std::vector<Geometric> geometrics;     /// ==> Macierz A, accumulative offset for each primitive


/// constraints register
static std::vector<Constraint> constraints;   /// ===> Wiezy , accumulative offset for each constraint


/// parameters register
static std::vector<Parameter> parameters;     /// paramLocation id-> param_offset


/// Point Locations in computation matrix [id] -> point offset
static int* poLocations;


/// Parameter locations [id] -> parameter offset
static int* paramLocations;



/// corespond to java implementations 
struct SolverStat {

    long startTime;
    long stopTime;
    long timeDelta;
    int size;
    int coefficientArity;
    int dimension;
    long accEvaluationTime;
    long accSolverTime;
    bool convergence;
    double error;
    double constraintDelta;
    int iterations;
};


static SolverStat stat;

/**
 * @brief  setup all matricies for computation and prepare kernel stream  intertwined with cusolver
 * 
 */
void solveSystemOnGPU();


/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    initDriver
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_initDriver
  (JNIEnv *env, jclass clazz)
{
    int count;

    error_t = cudaGetDeviceCount(&count);
    if(error_t != cudaSuccess) {
        printf("driver  error %d  = %s \n", static_cast<int>(error_t), cudaGetErrorString(error_t));
        return JNI_ERROR;
    }    

    deviceId = 0;

    error_t = cudaSetDevice(deviceId);
    if(error_t != cudaSuccess) {
        printf("driver  error %d  = %s \n", static_cast<int>(error_t), cudaGetErrorString(error_t));
        return JNI_ERROR;

    }
    return JNICALL;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    getLastError
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_mstruzek_jni_JNISolverGate_getLastError
  (JNIEnv *env, jclass clazz)
{
    const char* msg = cudaGetErrorString(error_t);
    return env->NewStringUTF(msg);
}


/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    closeDriver
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_closeDriver
  (JNIEnv *env, jclass clazz)
{
    error_t = cudaDeviceReset();
    if(error_t != cudaSuccess) {
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
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_resetComputationData
  (JNIEnv *env, jclass clazz)
{
    std::remove_if(points.begin(),      points.end(),       []() { return true; } );   
    std::remove_if(geometrics.begin(),  geometrics.end(),   []() { return true; } );
    std::remove_if(constraints.begin(), constraints.end(),  []() { return true; } );
    std::remove_if(parameters.begin(),  parameters.end(),   []() { return true; } );

    error_t = cudaFreeHost(poLocations);
    if(error_t != cudaSuccess) {
        printf("free mem error %d  = %s \n", static_cast<int>(error_t), cudaGetErrorString(error_t));
        return JNI_ERROR;
    }

    error_t = cudaFreeHost(paramLocations);
    if(error_t != cudaSuccess) {
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
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_resetComputationContext
  (JNIEnv *env, jclass clazz)
{
    /// workspace - zwolnic pamiec i wyzerowac wskazniki , \\cusolver
    return JNI_SUCCESS;
}


/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    initComputationContext
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_initComputationContext
  (JNIEnv *env, jclass clazz)
{

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
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_solveSystem
  (JNIEnv *env, jclass clazz)
{


    return JNI_SUCCESS;
}


/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    registerPointType
 * Signature: (IDD)I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_registerPointType
  (JNIEnv *env, jclass clazz, jint id, jdouble px, jdouble py)
{
    points.emplace_back(id, px, py);
    return JNI_SUCCESS;
}


/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    registerGeometricType
 * Signature: (IIIIIIII)I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_registerGeometricType
  (JNIEnv *env, jclass clazz, jint id, jint geometricTypeId, jint p1, jint p2, jint p3, jint a, jint b, jint c, jint d)
{      
    geometrics.emplace_back(id, geometricTypeId, p1, p2, p3, a, b, c, d);
    return JNI_SUCCESS;
}


/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    registerParameterType
 * Signature: (ID)I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_registerParameterType
  (JNIEnv *env, jclass clazz, jint id, jdouble value)
{
    parameters.emplace_back(id, value);
    return JNI_SUCCESS;
}


/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    registerConstraintType
 * Signature: (IIIIIIDD)I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_registerConstraintType
  (JNIEnv *env, jclass clazz, jint id, jint jconstraintTypeId, jint k, jint l, jint m, jint n, jint paramId, jdouble vecX, jdouble vecY)
{
    constraints.emplace_back(id, jconstraintTypeId, k, l, m, n, paramId, vecX, vecY);
    return JNI_SUCCESS;
}


/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    getPointPXCoordinate
 * Signature: (I)D
 */
JNIEXPORT jdouble JNICALL Java_com_mstruzek_jni_JNISolverGate_getPointPXCoordinate
  (JNIEnv *env, jclass clazz, jint id)
{
    int offset = poLocations[id];    
    double px = points[offset].px;    
    return (jdouble)px;
}


/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    getPointPYCoordinate
 * Signature: (I)D
 */
JNIEXPORT jdouble JNICALL Java_com_mstruzek_jni_JNISolverGate_getPointPYCoordinate
  (JNIEnv *env, jclass clazz, jint id)
{
    int offset = poLocations[id];    
    double py = points[offset].py;    
    return (jdouble)py;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    getPointCoordinateVector
 * Signature: ()[D
 */
JNIEXPORT jdoubleArray JNICALL Java_com_mstruzek_jni_JNISolverGate_getPointCoordinateVector
  (JNIEnv *env, jclass clazz)
  {
      /// depraceted or remove !
      return (jdoubleArray)NULL;
  }


/**
 *  
 * 
 */
void solveSystemOnGPU() {




}