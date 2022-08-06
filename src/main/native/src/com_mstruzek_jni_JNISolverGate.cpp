#include "com_mstruzek_jni_JNISolverGate.h"

#include <cuda_runtime_api.h>
#include <stdio.h>
#include <vector>
#include <algorithm>

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
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_initDriver(JNIEnv *env, jclass clazz)
{
  int count;

  error_t = cudaGetDeviceCount(&count);
  if (error_t != cudaSuccess)
  {
    printf("driver  error %d  = %s \n", static_cast<int>(error_t), cudaGetErrorString(error_t));
    return JNI_ERROR;
  }

  deviceId = 0;

  error_t = cudaSetDevice(deviceId);
  if (error_t != cudaSuccess)
  {
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
JNIEXPORT jstring JNICALL Java_com_mstruzek_jni_JNISolverGate_getLastError(JNIEnv *env, jclass clazz)
{
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
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_closeDriver(JNIEnv *env, jclass clazz)
{
  error_t = cudaDeviceReset();
  if (error_t != cudaSuccess)
  {
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
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_resetComputationData(JNIEnv *env, jclass clazz)
{
  std::remove_if(points.begin(), points.end(), [](auto _)
                 { return true; });
  std::remove_if(geometrics.begin(), geometrics.end(), [](auto _)
                 { return true; });
  std::remove_if(constraints.begin(), constraints.end(), [](auto _)
                 { return true; });
  std::remove_if(parameters.begin(), parameters.end(), [](auto _)
                 { return true; });

  error_t = cudaFreeHost(poLocations);
  if (error_t != cudaSuccess)
  {
    printf("free mem error %d  = %s \n", static_cast<int>(error_t), cudaGetErrorString(error_t));
    return JNI_ERROR;
  }

  error_t = cudaFreeHost(paramLocations);
  if (error_t != cudaSuccess)
  {
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
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_resetComputationContext(JNIEnv *env, jclass clazz)
{
  /// workspace - zwolnic pamiec i wyzerowac wskazniki , \\cusolver
  return JNI_SUCCESS;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    initComputationContext
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_initComputationContext(JNIEnv *env, jclass clazz)
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
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_solveSystem(JNIEnv *env, jclass clazz)
{

  int error;

  error = 0;

  solveSystemOnGPU(&error);

  if (error != 0)
  {
    return JNI_ERROR;
  }

  return JNI_SUCCESS;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    getSolverStatistics
 * Signature: ()Lcom/mstruzek/msketch/solver/SolverStat;
 */
JNIEXPORT jobject JNICALL Java_com_mstruzek_jni_JNISolverGate_getSolverStatistics(JNIEnv *env, jclass)
{
  jclass objClazz = env->FindClass("com/mstruzek/msketch/solver/SolverStat");
  if (objClazz == NULL)
  {
    printf("SolverStat is not visible in class loader\n");
    env->ThrowNew(env->FindClass("java/lang/RuntimeException"), "SolverStat is not visible in class loader");
    return 0;
  }

  jmethodID defaultCtor = env->GetMethodID(objClazz, "<init>", "()V");
  if (defaultCtor == NULL)
  {
    printf("constructor not visible\n");
    env->ThrowNew(env->FindClass("java/lang/RuntimeException"), "SolverStat constructor not visible");
    return 0;
  }

  jobject objStat = env->NewObject(objClazz, defaultCtor);
  if (objStat == NULL)
  {
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
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_registerPointType(JNIEnv *env, jclass clazz, jint id, jdouble px, jdouble py)
{
  points.emplace_back(id, px, py);
  return JNI_SUCCESS;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    registerGeometricType
 * Signature: (IIIIIIII)I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_registerGeometricType(JNIEnv *env, jclass clazz, jint id, jint geometricTypeId,
                                                                                 jint p1, jint p2, jint p3, jint a, jint b, jint c, jint d)
{
  geometrics.emplace_back(id, geometricTypeId, p1, p2, p3, a, b, c, d);
  return JNI_SUCCESS;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    registerParameterType
 * Signature: (ID)I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_registerParameterType(JNIEnv *env, jclass clazz, jint id, jdouble value)
{
  parameters.emplace_back(id, value);
  return JNI_SUCCESS;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    registerConstraintType
 * Signature: (IIIIIIDD)I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_registerConstraintType(JNIEnv *env, jclass clazz, jint id, jint jconstraintTypeId,
                                                                                  jint k, jint l, jint m, jint n, jint paramId, jdouble vecX, jdouble vecY)
{
  constraints.emplace_back(id, jconstraintTypeId, k, l, m, n, paramId, vecX, vecY);
  return JNI_SUCCESS;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    getPointPXCoordinate
 * Signature: (I)D
 */
JNIEXPORT jdouble JNICALL Java_com_mstruzek_jni_JNISolverGate_getPointPXCoordinate(JNIEnv *env, jclass clazz, jint id)
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
JNIEXPORT jdouble JNICALL Java_com_mstruzek_jni_JNISolverGate_getPointPYCoordinate(JNIEnv *env, jclass clazz, jint id)
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
JNIEXPORT jdoubleArray JNICALL Java_com_mstruzek_jni_JNISolverGate_getPointCoordinateVector(JNIEnv *env, jclass clazz)
{
  /// depraceted or remove !
  return (jdoubleArray)NULL;
}

/**
 *
 *
 */
void solveSystemOnGPU(int *error)
{



  /// (cublasError -> error)

  /// tranzlacja bledow z cudy
}