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
//    settings::get()->DEBUG_CHECK_ARG = true;
  //  settings::get()->DEBUG_SOLVER_CONVERGENCE = true;



    err = jni_initComputationContext(&env, eclass);

    


/// --signature: GeometricConstraintSolver  2009-2022
    /// --file-format: V1
    /// --file-name: C:\Users\micha\Documents\___model_91.n.gcm.cpp

    /// --definition-begin:

    err = jni_registerPointType(&env, eclass, 0, -7.121757e+02, 1.021186e+03);
    err = jni_registerPointType(&env, eclass, 1, -7.121757e+02, 1.421186e+03);
    err = jni_registerPointType(&env, eclass, 2, -7.121757e+02, 2.221186e+03);
    err = jni_registerPointType(&env, eclass, 3, -7.121757e+02, 2.621186e+03);
    err = jni_registerPointType(&env, eclass, 4, -2.336245e+03, 2.221186e+03);
    err = jni_registerPointType(&env, eclass, 5, -7.121757e+02, 2.221186e+03);
    err = jni_registerPointType(&env, eclass, 6, 2.535962e+03, 2.221186e+03);
    err = jni_registerPointType(&env, eclass, 7, 4.160031e+03, 2.221186e+03);
    err = jni_registerPointType(&env, eclass, 8, 1.746198e+03, 2.093574e+03);
    err = jni_registerPointType(&env, eclass, 9, 2.535962e+03, 2.221186e+03);
    err = jni_registerPointType(&env, eclass, 10, 2.930845e+03, 2.284993e+03);
    err = jni_registerPointType(&env, eclass, 11, 3.720609e+03, 2.412606e+03);
    err = jni_registerPointType(&env, eclass, 12, 2.535962e+03, 1.021186e+03);
    err = jni_registerPointType(&env, eclass, 13, 2.535962e+03, 1.421186e+03);
    err = jni_registerPointType(&env, eclass, 14, 2.535962e+03, 2.221186e+03);
    err = jni_registerPointType(&env, eclass, 15, 2.535962e+03, 2.621186e+03);
    err = jni_registerPointType(&env, eclass, 16, -2.336245e+03, 1.421186e+03);
    err = jni_registerPointType(&env, eclass, 17, -7.121757e+02, 1.421186e+03);
    err = jni_registerPointType(&env, eclass, 18, 2.535962e+03, 1.421186e+03);
    err = jni_registerPointType(&env, eclass, 19, 4.160031e+03, 1.421186e+03);
    err = jni_registerPointType(&env, eclass, 20, -1.511507e+03, 2.188290e+03);
    err = jni_registerPointType(&env, eclass, 21, -7.121757e+02, 2.221186e+03);
    err = jni_registerPointType(&env, eclass, 22, -3.125103e+02, 2.237635e+03);
    err = jni_registerPointType(&env, eclass, 23, 4.868205e+02, 2.270531e+03);

    err = jni_registerGeometricType(&env, eclass, 0, GEOMETRIC_TYPE_ID_LINE, 1, 2, -1, 0, 3, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 1, GEOMETRIC_TYPE_ID_LINE, 5, 6, -1, 4, 7, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 2, GEOMETRIC_TYPE_ID_CIRCLE, 9, 10, -1, 8, 11, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 3, GEOMETRIC_TYPE_ID_LINE, 13, 14, -1, 12, 15, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 4, GEOMETRIC_TYPE_ID_LINE, 17, 18, -1, 16, 19, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 5, GEOMETRIC_TYPE_ID_CIRCLE, 21, 22, -1, 20, 23, -1, -1);

    err = jni_registerParameterType(&env, eclass, 0, 4.000000e+02);
    err = jni_registerParameterType(&env, eclass, 1, 2.000000e+00);

    err = jni_registerConstraintType(&env, eclass, 0, CONSTRAINT_TYPE_ID_FIX_POINT, 0, -1, -1, -1, -1, -7.121757e+02,
                                     1.021186e+03);
    err = jni_registerConstraintType(&env, eclass, 1, CONSTRAINT_TYPE_ID_FIX_POINT, 3, -1, -1, -1, -1, -7.121757e+02,
                                     2.621186e+03);
    err = jni_registerConstraintType(&env, eclass, 2, CONSTRAINT_TYPE_ID_FIX_POINT, 4, -1, -1, -1, -1, -2.336245e+03,
                                     2.221186e+03);
    err = jni_registerConstraintType(&env, eclass, 3, CONSTRAINT_TYPE_ID_FIX_POINT, 7, -1, -1, -1, -1, 4.160031e+03,
                                     2.221186e+03);
    err = jni_registerConstraintType(&env, eclass, 4, CONSTRAINT_TYPE_ID_LINES_PERPENDICULAR, 1, 2, 5, 6, -1,
                                     0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 5, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 2, 5, -1, -1, -1,
                                     0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 6, CONSTRAINT_TYPE_ID_SET_HORIZONTAL, 6, 5, 0, 0, -1, 0.000000e+00,
                                     0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 7, CONSTRAINT_TYPE_ID_FIX_POINT, 8, -1, -1, -1, -1, 1.746198e+03,
                                     2.093574e+03);
    err = jni_registerConstraintType(&env, eclass, 8, CONSTRAINT_TYPE_ID_FIX_POINT, 11, -1, -1, -1, -1, 3.720609e+03,
                                     2.412606e+03);
    err = jni_registerConstraintType(&env, eclass, 9, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 6, 9, -1, -1, -1,
                                     0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 10, CONSTRAINT_TYPE_ID_FIX_POINT, 12, -1, -1, -1, -1, 2.535962e+03,
                                     1.021186e+03);
    err = jni_registerConstraintType(&env, eclass, 11, CONSTRAINT_TYPE_ID_FIX_POINT, 15, -1, -1, -1, -1, 2.535962e+03,
                                     2.621186e+03);
    err = jni_registerConstraintType(&env, eclass, 12, CONSTRAINT_TYPE_ID_FIX_POINT, 16, -1, -1, -1, -1, -2.336245e+03,
                                     1.421186e+03);
    err = jni_registerConstraintType(&env, eclass, 13, CONSTRAINT_TYPE_ID_FIX_POINT, 19, -1, -1, -1, -1, 4.160031e+03,
                                     1.421186e+03);
    err = jni_registerConstraintType(&env, eclass, 14, CONSTRAINT_TYPE_ID_LINES_PERPENDICULAR, 17, 18, 1, 2, -1,
                                     0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 15, CONSTRAINT_TYPE_ID_LINES_PERPENDICULAR, 18, 17, 13, 14, -1,
                                     0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 16, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 1, 17, -1, -1, -1,
                                     0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 17, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 18, 13, -1, -1, -1,
                                     0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 18, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 6, 14, -1, -1, -1,
                                     0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 19, CONSTRAINT_TYPE_ID_FIX_POINT, 20, -1, -1, -1, -1, -1.511507e+03,
                                     2.188290e+03);
    err = jni_registerConstraintType(&env, eclass, 20, CONSTRAINT_TYPE_ID_FIX_POINT, 23, -1, -1, -1, -1, 4.868205e+02,
                                     2.270531e+03);
    err = jni_registerConstraintType(&env, eclass, 21, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 2, 21, -1, -1, -1,
                                     0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 22, CONSTRAINT_TYPE_ID_EQUAL_LENGTH, 21, 22, 9, 10, -1, 0.000000e+00,
                                     0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 23, CONSTRAINT_TYPE_ID_DISTANCE_2_POINTS, 9, 10, -1, -1, 0,
                                     0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 24, CONSTRAINT_TYPE_ID_PARAMETRIZED_LENGTH, 9, 10, 14, 13, 1,
                                     0.000000e+00, 0.000000e+00);

    /// --definition-end: 


    err = jni_initComputation(&env, eclass);

    err = jni_solveSystem(&env, eclass);


    err = jni_solveSystem(&env, eclass);

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
