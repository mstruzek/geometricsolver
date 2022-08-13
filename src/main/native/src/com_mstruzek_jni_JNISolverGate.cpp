#include "com_mstruzek_jni_JNISolverGate.h"

#include <algorithm>
#include <chrono>
#include <cuda_runtime_api.h>
#include <functional>
#include <memory>
#include <numeric>
#include <stdio.h>
#include <type_traits>
#include <vector>

#include <string>

#include <string.h>

#include "model.cuh"
#include "stop_watch.h"

#include "geometric_solver.h"

/// GPU common variables

static int deviceId;
static cudaError_t error;

/// points register
static std::vector<graph::Point> points; /// poLocations id-> point_offset

/// geometricc register
static std::vector<graph::Geometric> geometrics; /// ==> Macierz A, accumulative offset for each primitive

/// constraints register
static std::vector<graph::Constraint> constraints; /// ===> Wiezy , accumulative offset for each constraint

/// parameters register
static std::vector<graph::Parameter> parameters; /// paramLocation id-> param_offset

static graph::SolverStat stat;

/// Point  Offset in computation matrix [id] -> point offset   ~~ Gather Vectors
static std::shared_ptr<int[]> pointOffset;
static std::shared_ptr<int[]> constraintOffset;
static std::shared_ptr<int[]> geometricOffset;

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    initDriver
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_mstruzek_jni_JNISolverGate_initDriver(JNIEnv *env, jclass clazz) {
        int count;

        error = cudaGetDeviceCount(&count);
        if (error != cudaSuccess) {
                printf("driver  error %d  = %s \n", static_cast<int>(error), cudaGetErrorString(error));
                return JNI_ERROR;
        }

        deviceId = 0;

        error = cudaSetDevice(deviceId);
        if (error != cudaSuccess) {
                printf("driver  error %d  = %s \n", static_cast<int>(error), cudaGetErrorString(error));
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
        
        cudaError_t error;       
        jstring result;
               
        /// !!! Do not reset Last Error into cudaSuccess
        error = cudaPeekAtLastError();

        if(error == cudaSuccess) {                               
                result = env->NewStringUTF("");
        } else {
                std::string message = std::string(cudaGetErrorName(error)) + " : " + std::string(cudaGetErrorString(error));        
                result = env->NewStringUTF(message.c_str());
        }
        
        return result;
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

        pointOffset = NULL;
        constraintOffset = NULL;
        geometricOffset = NULL;

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
        int err = 0;
        try {
                solveSystemOnGPU(
                        points, geometrics, constraints, parameters, pointOffset, constraintOffset, geometricOffset, 
                        &stat, 
                        &err);

        } catch (const std::exception &e) {
                env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
                return JNI_ERROR;
        }

        if (error != 0) {
                return JNI_ERROR;
        }

        return JNI_SUCCESS;
}

template <typename FieldType> void JniSetFieldValue(JNIEnv *env, jclass objClazz, jobject object, const char *fieldName, FieldType &sourceField) {
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
JNIEXPORT jobject JNICALL Java_com_mstruzek_jni_JNISolverGate_getSolverStatistics(JNIEnv *env, jclass) {
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
        int offset = pointOffset[id];
        double px = points[offset].x;
        return (jdouble)px;
}

/*
 * Class:     com_mstruzek_jni_JNISolverGate
 * Method:    getPointPYCoordinate
 * Signature: (I)D
 */
JNIEXPORT jdouble JNICALL Java_com_mstruzek_jni_JNISolverGate_getPointPYCoordinate(JNIEnv *env, jclass clazz, jint id) {
        int offset = pointOffset[id];
        double py = points[offset].y;
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
