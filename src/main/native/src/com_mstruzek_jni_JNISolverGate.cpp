#include "com_mstruzek_jni_JNISolverGate.h"

#include <algorithm>
#include <chrono>
#include <cuda_runtime_api.h>
#include <functional>
#include <memory>
#include <numeric>
#include <stdio.h>
#include <string.h>
#include <string>
#include <type_traits>
#include <vector>

#include "stop_watch.h"

#include "geometric_solver.h"

/// GPU common variables

static int deviceId;
static cudaError_t error;

/// <summary>
/// solver computation stat received from last computation
/// </summary>
static solver::SolverStat solverStat = {};

#define DEVICE_ID 0
/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    initDriver
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_initDriver(JNIEnv *env, jclass clazz)
{
    int count = 0;

    error = cudaGetDeviceCount(&count);
    if (error != cudaSuccess)
    {
        printf("driver  error %d  = %s \n", static_cast<int>(error), cudaGetErrorString(error));
        return JNI_ERROR;
    }

    if (count <= 0)
    {
        printf("cuda device not visible from driver %d  = %s \n", static_cast<int>(error), cudaGetErrorString(error));
        return JNI_ERROR;
    }

    ///  set first possible cuda device
    error = cudaSetDevice(DEVICE_ID);
    if (error != cudaSuccess)
    {
        printf("device driver  error, cuda-device [%d] ,  %d  = %s \n", DEVICE_ID, static_cast<int>(error),
               cudaGetErrorString(error));
        return JNI_ERROR;
    }

    printf("cuda device set to deviceId = [%d] \n", DEVICE_ID);
    return JNI_SUCCESS;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    getLastError
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_mstruzek_jni_JNISolverGate_getLastError(JNIEnv *env, jclass clazz)
{
    cudaError_t error = cudaPeekAtLastError(); /// !!! Do not reset Last Error into cudaSuccess
    jstring str = nullptr;
    if (error == cudaSuccess)
    {
        str = env->NewStringUTF("");
    }
    else
    {
        std::string message = std::string(cudaGetErrorName(error)) + " : " + std::string(cudaGetErrorString(error));
        str = env->NewStringUTF(message.c_str());
    }
    return str;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    closeDriver
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_closeDriver(JNIEnv *env, jclass clazz)
{
    error = cudaDeviceReset();
    if (error != cudaSuccess)
    {
        printf("driver  error %d  = %s \n", static_cast<int>(error), cudaGetErrorString(error));
        return JNI_ERROR;
    }
    printf("device reset completed ! \n");
    return JNI_SUCCESS;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    resetDatabase
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_resetComputationData(JNIEnv *env, jclass clazz)
{
    solver::resetComputationData(&error);
    if (error != cudaSuccess)
    {
        printf("[error] init computation data %d  = %s \n", static_cast<int>(error), cudaGetErrorString(error));
        return JNI_ERROR;
    }
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
    solver::resetComputationContext(&error);
    if (error != cudaSuccess)
    {
        printf("[error] reset computation context %d  = %s \n", static_cast<int>(error), cudaGetErrorString(error));
        return JNI_ERROR;
    }
    return JNI_SUCCESS;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    initComputationContext
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_initComputationContext(JNIEnv *env, jclass clazz)
{
    solver::initComputationContext(&error);
    if (error != cudaSuccess)
    {
        printf("[error] init computation context %d  = %s \n", static_cast<int>(error), cudaGetErrorString(error));
        return JNI_ERROR;
    }
    return JNI_SUCCESS;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    solveSystem
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_solveSystem(JNIEnv *env, jclass clazz)
{
    int err = 0;
    try
    {
        /// <summary>
        /// Initialize main computation on GPU - standard linearized Newton-Raphson method.
        /// </summary>
        solver::solveSystemOnGPU(&solverStat, &error);
    }
    catch (const std::exception &e) {
        printf("exception  = %s \n", e.what());
        // env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
        return JNI_ERROR;
    }

    if (error != cudaSuccess)
    {
        printf("[error] computation failed on GPU,  %d  = %s \n", static_cast<int>(error), cudaGetErrorString(error));
        return JNI_ERROR;
    }
    return JNI_SUCCESS;
}

template <typename FieldType>
void JniSetFieldValue(JNIEnv *env, jclass objClazz, jobject object, const char *const fieldName, FieldType sourceField)
{
    if constexpr (std::is_same<FieldType, double>::value)
    {
        env->SetDoubleField(object, env->GetFieldID(objClazz, fieldName, "D"), sourceField);
    }
    else if constexpr (std::is_same<FieldType, long>::value)
    {
        env->SetLongField(object, env->GetFieldID(objClazz, fieldName, "J"), sourceField);
    }
    else if constexpr (std::is_same<FieldType, int>::value)
    {
        env->SetIntField(object, env->GetFieldID(objClazz, fieldName, "I"), sourceField);
    }
    else if constexpr (std::is_same<FieldType, bool>::value)
    {
        env->SetBooleanField(object, env->GetFieldID(objClazz, fieldName, "Z"), sourceField);
    }
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    getSolverStatistics
 * Signature: ()Lcom/mstruzek/msketch/solver/SolverStat;
 */
JNIEXPORT jobject JNICALL Java_com_mstruzek_jni_JNISolverGate_getSolverStatistics(JNIEnv *env, jclass)
{
    jclass objectClazz = env->FindClass("com/mstruzek/msketch/solver/SolverStat");
    if (objectClazz == NULL)
    {
        printf("SolverStat is not visible in class loader\n");
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), "SolverStat is not visible in class loader");
        return 0;
    }

    jmethodID defaultCtor = env->GetMethodID(objectClazz, "<init>", "()V");
    if (defaultCtor == NULL)
    {
        printf("constructor not visible\n");
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), "SolverStat constructor not visible");
        return 0;
    }

    jobject solverStatObject = env->NewObject(objectClazz, defaultCtor);
    if (solverStatObject == NULL)
    {
        printf("object statt <init> error\n");
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), "construction instance error");
        return 0;
    }

    solver::SolverStat &stat = solverStat;

    JniSetFieldValue(env, objectClazz, solverStatObject, "startTime", stat.startTime);
    JniSetFieldValue(env, objectClazz, solverStatObject, "stopTime", stat.stopTime);
    JniSetFieldValue(env, objectClazz, solverStatObject, "timeDelta", stat.timeDelta);
    JniSetFieldValue(env, objectClazz, solverStatObject, "size", stat.size);
    JniSetFieldValue(env, objectClazz, solverStatObject, "coefficientArity", stat.coefficientArity);
    JniSetFieldValue(env, objectClazz, solverStatObject, "dimension", stat.dimension);
    JniSetFieldValue(env, objectClazz, solverStatObject, "accEvaluationTime", stat.accEvaluationTime);
    JniSetFieldValue(env, objectClazz, solverStatObject, "accSolverTime", stat.accSolverTime);
    JniSetFieldValue(env, objectClazz, solverStatObject, "convergence", stat.convergence);
    JniSetFieldValue(env, objectClazz, solverStatObject, "error", stat.error);
    JniSetFieldValue(env, objectClazz, solverStatObject, "constraintDelta", stat.constraintDelta);
    JniSetFieldValue(env, objectClazz, solverStatObject, "iterations", stat.iterations);

    return solverStatObject;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    registerPointType
 * Signature: (IDD)I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_registerPointType(JNIEnv *env, jclass clazz, jint id,
                                                                             jdouble px, jdouble py)
{
    solver::registerPointType(id, px, py);
    return JNI_SUCCESS;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    registerGeometricType
 * Signature: (IIIIIIII)I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_registerGeometricType(JNIEnv *env, jclass clazz, jint id,
                                                                                 jint geometricTypeId, jint p1, jint p2,
                                                                                 jint p3, jint a, jint b, jint c,
                                                                                 jint d)
{
    solver::registerGeometricType(id, geometricTypeId, p1, p2, p3, a, b, c, d);
    return JNI_SUCCESS;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    registerParameterType
 * Signature: (ID)I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_registerParameterType(JNIEnv *env, jclass clazz, jint id,
                                                                                 jdouble value)
{
    solver::registerParameterType(id, value);
    return JNI_SUCCESS;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    registerConstraintType
 * Signature: (IIIIIIDD)I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_registerConstraintType(JNIEnv *env, jclass clazz, jint id,
                                                                                  jint jconstraintTypeId, jint k,
                                                                                  jint l, jint m, jint n, jint paramId,
                                                                                  jdouble vecX, jdouble vecY)
{
    solver::registerConstraintType(id, jconstraintTypeId, k, l, m, n, paramId, vecX, vecY);
    return JNI_SUCCESS;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    getPointPXCoordinate
 * Signature: (I)D
 */
JNIEXPORT jdouble JNICALL Java_com_mstruzek_jni_JNISolverGate_getPointPXCoordinate(JNIEnv *env, jclass clazz, jint id)
{
    double px = solver::getPointPXCoordinate(id);
    return (jdouble)px;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    getPointPYCoordinate
 * Signature: (I)D
 */
JNIEXPORT jdouble JNICALL Java_com_mstruzek_jni_JNISolverGate_getPointPYCoordinate(JNIEnv *env, jclass clazz, jint id)
{
    double py = solver::getPointPYCoordinate(id);
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
    jboolean isCopy;
    /// przeinicjalizowac wypelenieni tego vecotr
    jdoubleArray stateVector = env->NewDoubleArray(2 * 1024);
    /// acquire pinne or copy
    jdouble *state_arr = env->GetDoubleArrayElements(stateVector, &isCopy);

    solver::getPointCoordinateVector(state_arr);

    if (isCopy == JNI_TRUE)
    {
        env->ReleaseDoubleArrayElements(stateVector, (double *)state_arr, 0);
    }
    else
    {
        env->ReleaseDoubleArrayElements(stateVector, (double *)state_arr, 0);
    }

    return stateVector;
}
