#include <iostream>
#include <stdio.h>
#include <cstddef>

#include "stop_watch.h"

#include "model_config.h"
#include "geometric_solver.h"

#include "com_mstruzek_jni_JNISolverGate.h"


#define jni_initDriver                 Java_com_mstruzek_jni_JNISolverGate_initDriver
#define jni_getLastError               Java_com_mstruzek_jni_JNISolverGate_getLastError
#define jni_closeDriver                Java_com_mstruzek_jni_JNISolverGate_closeDriver
#define jni_resetComputationData       Java_com_mstruzek_jni_JNISolverGate_resetComputationData
#define jni_resetComputationContext    Java_com_mstruzek_jni_JNISolverGate_resetComputationContext
#define jni_initComputationContext     Java_com_mstruzek_jni_JNISolverGate_initComputationContext
#define jni_solveSystem                Java_com_mstruzek_jni_JNISolverGate_solveSystem
#define jni_registerPointType          Java_com_mstruzek_jni_JNISolverGate_registerPointType
#define jni_registerGeometricType      Java_com_mstruzek_jni_JNISolverGate_registerGeometricType
#define jni_registerParameterType      Java_com_mstruzek_jni_JNISolverGate_registerParameterType
#define jni_registerConstraintType     Java_com_mstruzek_jni_JNISolverGate_registerConstraintType
#define jni_getPointPXCoordinate       Java_com_mstruzek_jni_JNISolverGate_getPointPXCoordinate
#define jni_getPointPYCoordinate       Java_com_mstruzek_jni_JNISolverGate_getPointPYCoordinate
#define jni_getPointCoordinateVector   Java_com_mstruzek_jni_JNISolverGate_getPointCoordinateVector


// JNI mock implementations
jclass FindClass_Impl(JNIEnv *env, const char *name);

// JNI mock implementations
jint ThrowNew_Impl(JNIEnv *env, jclass clazz, const char *msg);


int main(int argc, char* args[]) 
{
    int err;

    printf("empty inspector \n");

    ///  MOCK
    JNINativeInterface_ functions_iface = {};

    functions_iface.FindClass = FindClass_Impl;
    functions_iface.ThrowNew = ThrowNew_Impl;
        
    JNIEnv env = {};
    env.functions = &functions_iface;

    jclass eclass = nullptr;
        
    
    err = jni_initDriver(&env, eclass);

    err = jni_resetComputationData(&env, eclass);
    
    /// line_1
    err = jni_registerPointType(&env, eclass, 0, -20.0, 20.0);
    err = jni_registerPointType(&env, eclass, 1, 200.0, 20.0);
    err = jni_registerPointType(&env, eclass, 2, 600.0, 20.0);
    err = jni_registerPointType(&env, eclass, 3, 1000.0, 20.0);    
    err = jni_registerConstraintType(&env, eclass, 1, CONSTRAINT_TYPE_ID_FIX_POINT, 0, -1, -1, -1, -1, -20.0, 20.0);
    err = jni_registerConstraintType(&env, eclass, 2, CONSTRAINT_TYPE_ID_FIX_POINT, 3, -1, -1, -1, -1, 1000.0, 20.0);    
    err = jni_registerGeometricType(&env, eclass, 1, GEOMETRIC_TYPE_ID_LINE, 1, 2, -1, 0,  3,  -1, -1);
    
                                                                            // jint p1, jint p2, jint p3, jint a, jint b, jint c,jint d

    /// line_2
    err = jni_registerPointType(&env, eclass, 4, 20.0, 0.0);
    err = jni_registerPointType(&env, eclass, 5, 20.0, 210.0);
    err = jni_registerPointType(&env, eclass, 6, 20.0, 600.0);
    err = jni_registerPointType(&env, eclass, 7, 20.0, 800.0);    
    err = jni_registerConstraintType(&env, eclass, 3, CONSTRAINT_TYPE_ID_FIX_POINT, 4, -1, -1, -1, -1, 20.0, 0.0);
    err = jni_registerConstraintType(&env, eclass, 4, CONSTRAINT_TYPE_ID_FIX_POINT, 7, -1, -1, -1, -1, 20.0, 800.0);    
    err = jni_registerGeometricType(&env, eclass, 2, GEOMETRIC_TYPE_ID_LINE, 5, 6, 4, 7, -1, -1, -1);
   
    /// constraint(1,5)(Connect2Points)
    err = jni_registerConstraintType(&env, eclass, 5, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 1, 5, -1, -1, -1, -1, -1);    


    err = jni_initComputationContext(&env, eclass);


    err = jni_solveSystem(&env, eclass);


    return 0;
}

jclass FindClass_Impl(JNIEnv *env, const char *name)
{
    return NULL;
}

jint ThrowNew_Impl(JNIEnv *env, jclass clazz, const char *msg)
{
    throw new std::logic_error(msg);
}
