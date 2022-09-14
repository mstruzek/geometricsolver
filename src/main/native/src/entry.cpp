#include <iostream>
#include <stdio.h>
#include <cstddef>

#include "settings.h"
#include "cuerror.h"
#include "stop_watch.h"

#include "model_config.h"

#include "com_mstruzek_jni_JNISolverGate.h"


#define jni_initDriver                 Java_com_mstruzek_jni_JNISolverGate_initDriver
#define jni_setBooleanProperty         Java_com_mstruzek_jni_JNISolverGate_setBooleanProperty
#define jni_setLongProperty            Java_com_mstruzek_jni_JNISolverGate_setLongProperty
#define jni_getLastError               Java_com_mstruzek_jni_JNISolverGate_getLastError
#define jni_closeDriver                Java_com_mstruzek_jni_JNISolverGate_closeDriver
#define jni_initComputationContext     Java_com_mstruzek_jni_JNISolverGate_initComputationContext
#define jni_initComputation            Java_com_mstruzek_jni_JNISolverGate_initComputation
#define jni_solveSystem                Java_com_mstruzek_jni_JNISolverGate_solveSystem
#define jni_registerPointType          Java_com_mstruzek_jni_JNISolverGate_registerPointType
#define jni_registerGeometricType      Java_com_mstruzek_jni_JNISolverGate_registerGeometricType
#define jni_registerParameterType      Java_com_mstruzek_jni_JNISolverGate_registerParameterType
#define jni_registerConstraintType     Java_com_mstruzek_jni_JNISolverGate_registerConstraintType
#define jni_getPointPXCoordinate       Java_com_mstruzek_jni_JNISolverGate_getPointPXCoordinate
#define jni_getPointPYCoordinate       Java_com_mstruzek_jni_JNISolverGate_getPointPYCoordinate

#define jni_destroyComputation         Java_com_mstruzek_jni_JNISolverGate_destroyComputation
#define jni_destroyComputationContext  Java_com_mstruzek_jni_JNISolverGate_destroyComputationContext
#define jni__closeDriver               Java_com_mstruzek_jni_JNISolverGate_closeDriver



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
        
    
    err = jni_initDriver(&env, eclass, 0);

    //jni_setBooleanProperty  
    
        
    jni_setLongProperty(&env, eclass, 12, (jlong) 3);  // settings::get()->COMPUTATION_MODE  = /// 12  - 1 - DENSE_LAYOUT , 2 - SPARSE_LAYOUT , *3 - DIRECT_LAYOUT


    jni_setBooleanProperty(&env, eclass, 8, (jboolean) true);  // settings::get()->DEBUG_SOLVER_CONVERGENCE = false; 
    jni_setBooleanProperty(&env, eclass, 9, (jboolean) true);  // settings::get()->DEBUG_CHECK_ARG = false;
    jni_setBooleanProperty(&env, eclass, 2, (jboolean) true);  // settings::get()->DEBUG_TENSOR_A= true;
    jni_setBooleanProperty(&env, eclass, 3, (jboolean) true);  // settings::get()->DEBUG_TENSOR_B= true;

    jni_setBooleanProperty(&env, eclass, 7, (jboolean) true);  // settings::get()->SOLVER_INC_HESSIAN = false;
    
    

    err = jni_initComputationContext(&env, eclass);

    
    /// --signature: GeometricConstraintSolver  2009-2022
    /// --file-format: V1
    /// --file-name: e:\source\gsketcher\data\_vertical_line.cpp

    /// --definition-begin:

    err = jni_registerPointType(&env, eclass, 0, 1.655000e+02, 2.285000e+02);
    err = jni_registerPointType(&env, eclass, 1, 2.240000e+02, 2.420000e+02);
    err = jni_registerPointType(&env, eclass, 2, 3.410000e+02, 2.690000e+02);
    err = jni_registerPointType(&env, eclass, 3, 3.995000e+02, 2.825000e+02);

    err = jni_registerGeometricType(&env, eclass, 0, GEOMETRIC_TYPE_ID_LINE, 1, 2, -1, 0, 3, -1, -1);

    err = jni_registerConstraintType(&env, eclass, 0, CONSTRAINT_TYPE_ID_FIX_POINT, 0, -1, -1, -1, -1, 1.655000e+02,
                                     2.285000e+02);
    err = jni_registerConstraintType(&env, eclass, 1, CONSTRAINT_TYPE_ID_FIX_POINT, 3, -1, -1, -1, -1, 3.995000e+02,
                                     2.825000e+02);
    err = jni_registerConstraintType(&env, eclass, 2, CONSTRAINT_TYPE_ID_SET_VERTICAL, 1, 2, 0, 0, -1, 0.000000e+00,
                                     0.000000e+00);

    /// --definition-end: 


    err = jni_initComputation(&env, eclass);

    err = jni_solveSystem(&env, eclass);

    //err = jni_solveSystem(&env, eclass);

    try {

        err = jni_destroyComputation(&env, eclass);

        err = jni_destroyComputationContext(&env, eclass);
        err = jni_closeDriver(&env, eclass);


    } catch (std::exception &e) {
    
        printf("eee %s", e.what());
    }



    return 0;
}

jclass FindClass_Impl(JNIEnv *env, const char *name)
{
    return NULL;
}

jint ThrowNew_Impl(JNIEnv *env, jclass clazz, const char *msg)
{
    throw std::logic_error(msg);
}
