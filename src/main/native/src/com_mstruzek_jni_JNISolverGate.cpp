#include "com_mstruzek_jni_JNISolverGate.h"

#define JNI_SUCCESS com_mstruzek_jni_JNISolverGate_JNI_SUCCESS
#define JNI_ERROR com_mstruzek_jni_JNISolverGate_JNI_ERROR

#include <cuda_runtime_api.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

#include "cuerror.h"
#include "geometric_solver.h"
#include "settings.h"
#include "stop_watch.h"

/// GPU common variables

/// observed error from solver context
static cudaError_t error;

/// solver computation stat received from last computation
static solver::SolverStat solverStat = {};

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    getLastError
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_mstruzek_jni_JNISolverGate_getLastError(JNIEnv *env, jclass clazz) {
    cudaError_t error = cudaPeekAtLastError(); /// !!! Do not reset Last Error into cudaSuccess
    jstring str = nullptr;
    if (error == cudaSuccess) {
        str = env->NewStringUTF("");
    } else {
        std::string message = std::string(cudaGetErrorName(error)) + " : " + std::string(cudaGetErrorString(error));
        str = env->NewStringUTF(message.c_str());
    }
    return str;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    setBooleanProperty
 * Signature: (IZ)I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_setBooleanProperty(JNIEnv *env, jclass clazz, jint id,
                                                                              jboolean value) {
    int err = settings::setBooleanProperty(id, value);
    if (err != 0) {
        printf("[settings] bool property unrecognized,  id (%d) value (%d)", id, value);
        return JNI_ERROR;
    }
    return JNI_SUCCESS;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    setLongProperty
 * Signature: (IJ)I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_setLongProperty(JNIEnv *evn, jclass clazz, jint id,
                                                                           jlong value) {
    int err = settings::setLongProperty(id, static_cast<long>(value));
    if (err != 0) {
        printf("[settings] long property unrecognized,  id (%d) value (%zd)", id, value);
        return JNI_ERROR;
    }
    return JNI_SUCCESS;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    setDoubleProperty
 * Signature: (ID)I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_setDoubleProperty(JNIEnv *evn, jclass clazz, jint id,
                                                                             jdouble value) {
    int err = settings::setDoubleProperty(id, value);
    if (err != 0) {
        printf("[settings] double property unrecognized,  id (%d) value (%e)", id, value);
        return JNI_ERROR;
    }
    return JNI_SUCCESS;
}

std::string byteHashToString(char bytest[16]) {
    static const char ascii[16] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
    std::string result;
    std::for_each(bytest, bytest + 16, [&](char e) {
        char half1 = e & 0x0f;
        char half2 = e >> 4 & 0x0f;
        result.push_back(ascii[half2]);
        result.push_back(ascii[half1]);
    });
    return result;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    initDriver
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_initDriver(JNIEnv *env, jclass clazz, jint deviceId) {
    /// int DeviceId = 0;
    int count = 0;

    error = cudaGetDeviceCount(&count);
    if (error != cudaSuccess) {
        printf("driver  error %d  = %s \n", static_cast<int>(error), cudaGetErrorString(error));
        return JNI_ERROR;
    }
    if (count <= 0) {
        printf("cuda device not visible from driver %d  = %s \n", static_cast<int>(error), cudaGetErrorString(error));
        return JNI_ERROR;
    }

    /// wydrukujmy wszystkie mozliwe urzadzenia !!!
    int id = 0;
    while (id < count) {
        cudaDeviceProp deviceProp;

        error = cudaGetDeviceProperties(&deviceProp, id);
        if (error != cudaSuccess) {
            const char *errorStr = cudaGetErrorString(error);
            printf("device properties error, cuda-device [%d] ,  %d  = %s \n", id, static_cast<int>(error), errorStr);
            ++id;
            continue;
        }

        printf(" - =============================================== - \n");
        printf(" #deviceId              = ( %d ) \n", id);
        printf(" - uuid                 = %s \n", byteHashToString(deviceProp.uuid.bytes).c_str());
        printf(" - name                 = %s \n", deviceProp.name);
        printf(" - totalGlobalMem       = %zd \n", deviceProp.totalGlobalMem);
        printf(" - regsPerBlock         = %d \n", deviceProp.regsPerBlock);
        printf(" - warpSize             = %d \n", deviceProp.warpSize);
        printf(" - maxThreadsPerBlock   = %d \n", deviceProp.maxThreadsPerBlock);
        printf(" - maxThreadsDim[3];    = (%d, %d, %d) \n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf(" - maxGridSize[3];      = (%d, %d, %d) \n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);

        printf(" - major                  = %d \n", deviceProp.major);
        printf(" - minor                  = %d \n", deviceProp.minor);
        printf(" - multiProcessorCount    = %d \n", deviceProp.multiProcessorCount);
        printf(" - computeMode            = %d \n", deviceProp.computeMode);
        printf(" - concurrentKernels      = %d \n", deviceProp.concurrentKernels);
        printf(" - ECCEnabled             = %d \n", deviceProp.ECCEnabled);
        printf(" - pciBusID               = %d \n", deviceProp.pciBusID);
        printf(" - pciDeviceID            = %d \n", deviceProp.pciDeviceID);
        printf(" - pciDomainID            = %d \n", deviceProp.pciDomainID);
        printf(" - asyncEngineCount       = %d \n", deviceProp.asyncEngineCount);
        printf(" - l2CacheSize            = %d \n", deviceProp.l2CacheSize);
        printf(" - regsPerMultiprocessor  = %d \n", deviceProp.regsPerMultiprocessor);

        ++id;
    }

    ///  set first possible cuda device
    error = cudaSetDevice(deviceId);
    if (error != cudaSuccess) {
        printf("device driver  error, cuda-device [%d] ,  %d  = %s \n", deviceId, static_cast<int>(error),
               cudaGetErrorString(error));
        return JNI_ERROR;
    }

    printf("cuda device set to deviceId = [%d] \n", deviceId);
    return JNI_SUCCESS;
}

///============================================================
/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    initComputation
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_initComputationContext(JNIEnv *env, jclass clazz) {
    solver::initComputationContext(&error);
    if (error != cudaSuccess) {
        printf("[error] init computation data %d  = %s \n", static_cast<int>(error), cudaGetErrorString(error));
        return JNI_ERROR;
    }
    return JNI_SUCCESS;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    getCommitTime
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_mstruzek_jni_JNISolverGate_getCommitTime(JNIEnv *env, jclass clazz) {
    return (jlong)solver::getCommitTime();
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    registerPointType
 * Signature: (IDD)I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_registerPointType(JNIEnv *env, jclass clazz, jint id,
                                                                             jdouble px, jdouble py) {
    solver::registerPointType(id, px, py);
    return JNI_SUCCESS;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    registerGeometricType
 * Signature: (IIIIIIIII)I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_registerGeometricType(JNIEnv *env, jclass clazz, jint id,
                                                                                 jint geometricTypeId, jint p1, jint p2,
                                                                                 jint p3, jint a, jint b, jint c,
                                                                                 jint d) {
    solver::registerGeometricType(id, geometricTypeId, p1, p2, p3, a, b, c, d);
    return JNI_SUCCESS;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    registerParameterType
 * Signature: (ID)I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_registerParameterType(JNIEnv *env, jclass clazz, jint id,
                                                                                 jdouble value) {
    solver::registerParameterType(id, value);
    return JNI_SUCCESS;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    registerConstraintType
 * Signature: (IIIIIIIDD)I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_registerConstraintType(JNIEnv *env, jclass clazz, jint id,
                                                                                  jint jconstraintTypeId, jint k,
                                                                                  jint l, jint m, jint n, jint paramId,
                                                                                  jdouble vecX, jdouble vecY) {
    solver::registerConstraintType(id, jconstraintTypeId, k, l, m, n, paramId, vecX, vecY);
    return JNI_SUCCESS;
}

/// ===================================================================================================================

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    initComputation
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_initComputation(JNIEnv *env, jclass clazz) {
    solver::initComputation(&error);
    if (error != cudaSuccess) {
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
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_solveSystem(JNIEnv *env, jclass clazz) {
    int err = 0;
    try {
        /// <summary>
        /// Initialize main computation on GPU - standard linearized Newton-Raphson
        /// method.
        /// </summary>
        solver::solveSystemOnGPU(&solverStat, &error);
    } catch (const std::exception &e) {
        printf("[error] exception  = %s \n", e.what());
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
        return JNI_ERROR;
    }

    if (error != cudaSuccess) {
        printf("[error] computation failed on GPU,  %d  = %s \n", static_cast<int>(error), cudaGetErrorString(error));
        return JNI_ERROR;
    }
    return JNI_SUCCESS;
}

/// ===================================================================================================================

template <typename FieldType>
void JniSetFieldValue(JNIEnv *env, jclass objClazz, jobject object, const char *const fieldName,
                      FieldType sourceField) {
    if constexpr (std::is_same<FieldType, double>::value) {
        env->SetDoubleField(object, env->GetFieldID(objClazz, fieldName, "D"), sourceField);
    } else if constexpr (std::is_same<FieldType, long>::value) {
        env->SetLongField(object, env->GetFieldID(objClazz, fieldName, "J"), sourceField);
    } else if constexpr (std::is_same<FieldType, int>::value) {
        env->SetIntField(object, env->GetFieldID(objClazz, fieldName, "I"), sourceField);
    } else if constexpr (std::is_same<FieldType, bool>::value) {
        env->SetBooleanField(object, env->GetFieldID(objClazz, fieldName, "Z"), sourceField);
    }
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    getSolverStatistics
 * Signature: ()Lcom/mstruzek/msketch/solver/SolverStat;
 */
JNIEXPORT jobject JNICALL Java_com_mstruzek_jni_JNISolverGate_getSolverStatistics(JNIEnv *env, jclass clazz) {
    jclass objectClazz = env->FindClass("com/mstruzek/msketch/solver/SolverStat");
    if (objectClazz == NULL) {
        printf("SolverStat is not visible in class loader\n");
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), "SolverStat is not visible in class loader");
        return 0;
    }

    jmethodID defaultCtor = env->GetMethodID(objectClazz, "<init>", "()V");
    if (defaultCtor == NULL) {
        printf("constructor not visible\n");
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), "SolverStat constructor not visible");
        return 0;
    }

    jobject solverStatObject = env->NewObject(objectClazz, defaultCtor);
    if (solverStatObject == NULL) {
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
 * Method:    fetchStateVector
 * Signature: ()[D
 */
JNIEXPORT jdoubleArray JNICALL Java_com_mstruzek_jni_JNISolverGate_fetchStateVector(JNIEnv *env, jclass clazz) {
    jboolean isCopy;
    solver::SolverStat &stat = solverStat;
    /// przeinicjalizowac wypelenieni tego vecotr !!

    jdoubleArray jStateVector = env->NewDoubleArray(stat.size);

    /// acquire pined or copy
    jdouble *stateVector = env->GetDoubleArrayElements(jStateVector, &isCopy);

    /// update current state
    solver::fillPointCoordinateVector(stateVector);

    if (isCopy == JNI_TRUE) {
        env->ReleaseDoubleArrayElements(jStateVector, (double *)stateVector, 0);
    }

    jStateVector = (jdoubleArray)env->NewGlobalRef(jStateVector);
    if (jStateVector == nullptr) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      "[global reference] double array  initializatio error");
        return nullptr;
    }
    return jStateVector;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    updateStateVector
 * Signature: ([D)I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_updateStateVector(JNIEnv *env, jclass clazz,
                                                                             jdoubleArray jStateVector) {

    jboolean isCopy;
    /// acquire pined or copy
    jdouble *stateVector = env->GetDoubleArrayElements(jStateVector, &isCopy);

    /// update current state
    solver::updatePointCoordinateVector(stateVector);

    if (isCopy == JNI_TRUE) {
        env->ReleaseDoubleArrayElements(jStateVector, (double *)stateVector, 0);
    }
    return JNI_SUCCESS;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    updateConstraintState
 * Signature: (IDD)I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_updateConstraintState(JNIEnv *env, jclass clazz,
                                                                                 jint constraintId, jdouble vecX,
                                                                                 jdouble vecY) {
    int err = solver::updateConstraintState(constraintId, vecX, vecY);
    if (err != 0) {
        return JNI_ERROR;
    }
    return JNI_SUCCESS;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    getPointPXCoordinate
 * Signature: (I)D
 */
JNIEXPORT jdouble JNICALL Java_com_mstruzek_jni_JNISolverGate_getPointPXCoordinate(JNIEnv *env, jclass clazz, jint id) {
    double px = solver::getPointPXCoordinate(id);
    return (jdouble)px;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    getPointPYCoordinate
 * Signature: (I)D
 */
JNIEXPORT jdouble JNICALL Java_com_mstruzek_jni_JNISolverGate_getPointPYCoordinate(JNIEnv *env, jclass clazz, jint id) {
    double py = solver::getPointPYCoordinate(id);
    return (jdouble)py;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    destroyComputation
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_destroyComputation(JNIEnv *env, jclass clazz) {
    solver::destroyComputation(&error);
    if (error != cudaSuccess) {
        printf("[error] init computation data %d  = %s \n", static_cast<int>(error), cudaGetErrorString(error));
        return JNI_ERROR;
    }
    return JNI_SUCCESS;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    destroyComputationContext
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_destroyComputationContext(JNIEnv *env, jclass clazz) {
    /// workspace - zwolnic pamiec i wyzerowac wskazniki , \\cusolver
    solver::destroyComputationContext(&error);
    if (error != cudaSuccess) {
        printf("[error] reset computation context %d  = %s \n", static_cast<int>(error), cudaGetErrorString(error));
        return JNI_ERROR;
    }
    return JNI_SUCCESS;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    closeDriver
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_closeDriver(JNIEnv *env, jclass clazz) {
    error = cudaDeviceReset();
    if (error != cudaSuccess) {
        printf("driver  error %d  = %s \n", static_cast<int>(error), cudaGetErrorString(error));
        return JNI_ERROR;
    }
    printf("device reset completed ! \n");
    return JNI_SUCCESS;
}

/// ===================================================================================================================
/// ///
