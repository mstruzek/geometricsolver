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


int model_single_line(JNIEnv &env, jclass &eclass) {
    int err;
    /// --signature: GeometricConstraintSolver  2009-2022
    /// --file-format: V1
    /// --file-name: e:\source\gsketcher\data\_vertical_line.cpp

    /// --definition-begin:

    err = jni_registerPointType(&env, eclass, 0, 1.655000e+02, 2.285000e+02);
    err = jni_registerPointType(&env, eclass, 1, 2.240000e+02, 2.420000e+02);
    err = jni_registerPointType(&env, eclass, 2, 3.410000e+02, 2.690000e+02);
    err = jni_registerPointType(&env, eclass, 3, 3.995000e+02, 2.825000e+02);

    err = jni_registerGeometricType(&env, eclass, 0, GEOMETRIC_TYPE_ID_LINE, 1, 2, -1, 0, 3, -1, -1);

    err = jni_registerConstraintType(&env, eclass, 0, CONSTRAINT_TYPE_ID_FIX_POINT, 0, -1, -1, -1, -1, 1.655000e+02, 2.285000e+02);
    err = jni_registerConstraintType(&env, eclass, 1, CONSTRAINT_TYPE_ID_FIX_POINT, 3, -1, -1, -1, -1, 3.995000e+02, 2.825000e+02);
    err = jni_registerConstraintType(&env, eclass, 2, CONSTRAINT_TYPE_ID_SET_VERTICAL, 1, 2, 0, 0, -1, 0.000000e+00, 0.000000e+00);

    /// --definition-end:
    return err;
}


int model_circle_line_tangetn_perpendicular(JNIEnv &env, jclass& eclass) {

    /// --signature: GeometricConstraintSolver  2009-2022
    /// --file-format: V1
    /// --file-name: e:\source\gsketcher\data\_cirle_tangent_3.cpp

    /// --definition-begin:
    int err;
    err = jni_registerPointType(&env, eclass, 0, 1.122606e+02, -5.216158e+00);
    err = jni_registerPointType(&env, eclass, 1, 1.121829e+02, 1.050995e+02);
    err = jni_registerPointType(&env, eclass, 2, 1.140000e+02, 2.486018e+02);
    err = jni_registerPointType(&env, eclass, 3, 1.145798e+02, 4.145594e+02);
    err = jni_registerPointType(&env, eclass, 4, 5.336543e+01, 3.595768e+02);
    err = jni_registerPointType(&env, eclass, 5, 2.875641e+02, 1.877037e+02);
    err = jni_registerPointType(&env, eclass, 6, 3.710000e+02, 1.790000e+02);
    err = jni_registerPointType(&env, eclass, 7, 7.110897e+02, 2.152821e+02);

    err = jni_registerGeometricType(&env, eclass, 0, GEOMETRIC_TYPE_ID_LINE, 1, 2, -1, 0, 3, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 1, GEOMETRIC_TYPE_ID_CIRCLE, 5, 6, -1, 4, 7, -1, -1);

    err = jni_registerConstraintType(&env, eclass, 0, CONSTRAINT_TYPE_ID_FIX_POINT, 0, -1, -1, -1, -1, 1.122606e+02, -5.216158e+00);
    err = jni_registerConstraintType(&env, eclass, 1, CONSTRAINT_TYPE_ID_FIX_POINT, 3, -1, -1, -1, -1, 1.145798e+02, 4.145594e+02);
    err = jni_registerConstraintType(&env, eclass, 2, CONSTRAINT_TYPE_ID_FIX_POINT, 4, -1, -1, -1, -1, 5.336543e+01, 3.595768e+02);
    err = jni_registerConstraintType(&env, eclass, 3, CONSTRAINT_TYPE_ID_FIX_POINT, 7, -1, -1, -1, -1, 7.110897e+02, 2.152821e+02);
    err = jni_registerConstraintType(&env, eclass, 4, CONSTRAINT_TYPE_ID_FIX_POINT, 1, -1, -1, -1, -1, 1.140000e+02, 1.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 5, CONSTRAINT_TYPE_ID_SET_VERTICAL, 1, 2, 0, 0, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 6, CONSTRAINT_TYPE_ID_TANGENCY, 1, 2, 5, 6, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 7, CONSTRAINT_TYPE_ID_LINES_PERPENDICULAR, 1, 2, 5, 6, -1, 0.000000e+00, 0.000000e+00);

    /// --definition-end:

    return err;
}

int model_6x10(JNIEnv &env, jclass &eclass) {

    int err;

    /// --signature: GeometricConstraintSolver  2009-2022
    /// --file-format: V1
    /// --file-name: e:\source\gsketcher\data\__square_nets_1768_6x10.gcm.cpp

    /// --definition-begin:

    err = jni_registerPointType(&env, eclass, 0, 0.000000e+00, -7.500000e+01);
    err = jni_registerPointType(&env, eclass, 1, 0.000000e+00, -5.000000e+01);
    err = jni_registerPointType(&env, eclass, 2, 0.000000e+00, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 3, 0.000000e+00, 2.500000e+01);
    err = jni_registerPointType(&env, eclass, 4, -7.500000e+01, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 5, -5.000000e+01, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 6, 0.000000e+00, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 7, 2.500000e+01, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 8, 5.000000e+01, -7.500000e+01);
    err = jni_registerPointType(&env, eclass, 9, 5.000000e+01, -5.000000e+01);
    err = jni_registerPointType(&env, eclass, 10, 5.000000e+01, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 11, 5.000000e+01, 2.500000e+01);
    err = jni_registerPointType(&env, eclass, 12, -2.500000e+01, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 13, 0.000000e+00, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 14, 5.000000e+01, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 15, 7.500000e+01, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 16, 1.000000e+02, -7.500000e+01);
    err = jni_registerPointType(&env, eclass, 17, 1.000000e+02, -5.000000e+01);
    err = jni_registerPointType(&env, eclass, 18, 1.000000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 19, 1.000000e+02, 2.500000e+01);
    err = jni_registerPointType(&env, eclass, 20, 2.500000e+01, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 21, 5.000000e+01, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 22, 1.000000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 23, 1.250000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 24, 1.500000e+02, -7.500000e+01);
    err = jni_registerPointType(&env, eclass, 25, 1.500000e+02, -5.000000e+01);
    err = jni_registerPointType(&env, eclass, 26, 1.500000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 27, 1.500000e+02, 2.500000e+01);
    err = jni_registerPointType(&env, eclass, 28, 7.500000e+01, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 29, 1.000000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 30, 1.500000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 31, 1.750000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 32, 2.000000e+02, -7.500000e+01);
    err = jni_registerPointType(&env, eclass, 33, 2.000000e+02, -5.000000e+01);
    err = jni_registerPointType(&env, eclass, 34, 2.000000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 35, 2.000000e+02, 2.500000e+01);
    err = jni_registerPointType(&env, eclass, 36, 1.250000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 37, 1.500000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 38, 2.000000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 39, 2.250000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 40, 2.500000e+02, -7.500000e+01);
    err = jni_registerPointType(&env, eclass, 41, 2.500000e+02, -5.000000e+01);
    err = jni_registerPointType(&env, eclass, 42, 2.500000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 43, 2.500000e+02, 2.500000e+01);
    err = jni_registerPointType(&env, eclass, 44, 1.750000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 45, 2.000000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 46, 2.500000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 47, 2.750000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 48, 3.000000e+02, -7.500000e+01);
    err = jni_registerPointType(&env, eclass, 49, 3.000000e+02, -5.000000e+01);
    err = jni_registerPointType(&env, eclass, 50, 3.000000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 51, 3.000000e+02, 2.500000e+01);
    err = jni_registerPointType(&env, eclass, 52, 2.250000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 53, 2.500000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 54, 3.000000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 55, 3.250000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 56, 3.500000e+02, -7.500000e+01);
    err = jni_registerPointType(&env, eclass, 57, 3.500000e+02, -5.000000e+01);
    err = jni_registerPointType(&env, eclass, 58, 3.500000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 59, 3.500000e+02, 2.500000e+01);
    err = jni_registerPointType(&env, eclass, 60, 2.750000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 61, 3.000000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 62, 3.500000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 63, 3.750000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 64, 4.000000e+02, -7.500000e+01);
    err = jni_registerPointType(&env, eclass, 65, 4.000000e+02, -5.000000e+01);
    err = jni_registerPointType(&env, eclass, 66, 4.000000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 67, 4.000000e+02, 2.500000e+01);
    err = jni_registerPointType(&env, eclass, 68, 3.250000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 69, 3.500000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 70, 4.000000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 71, 4.250000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 72, 4.500000e+02, -7.500000e+01);
    err = jni_registerPointType(&env, eclass, 73, 4.500000e+02, -5.000000e+01);
    err = jni_registerPointType(&env, eclass, 74, 4.500000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 75, 4.500000e+02, 2.500000e+01);
    err = jni_registerPointType(&env, eclass, 76, 3.750000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 77, 4.000000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 78, 4.500000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 79, 4.750000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 80, 0.000000e+00, -2.500000e+01);
    err = jni_registerPointType(&env, eclass, 81, 0.000000e+00, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 82, 0.000000e+00, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 83, 0.000000e+00, 7.500000e+01);
    err = jni_registerPointType(&env, eclass, 84, -7.500000e+01, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 85, -5.000000e+01, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 86, 0.000000e+00, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 87, 2.500000e+01, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 88, 5.000000e+01, -2.500000e+01);
    err = jni_registerPointType(&env, eclass, 89, 5.000000e+01, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 90, 5.000000e+01, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 91, 5.000000e+01, 7.500000e+01);
    err = jni_registerPointType(&env, eclass, 92, -2.500000e+01, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 93, 0.000000e+00, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 94, 5.000000e+01, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 95, 7.500000e+01, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 96, 1.000000e+02, -2.500000e+01);
    err = jni_registerPointType(&env, eclass, 97, 1.000000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 98, 1.000000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 99, 1.000000e+02, 7.500000e+01);
    err = jni_registerPointType(&env, eclass, 100, 2.500000e+01, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 101, 5.000000e+01, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 102, 1.000000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 103, 1.250000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 104, 1.500000e+02, -2.500000e+01);
    err = jni_registerPointType(&env, eclass, 105, 1.500000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 106, 1.500000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 107, 1.500000e+02, 7.500000e+01);
    err = jni_registerPointType(&env, eclass, 108, 7.500000e+01, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 109, 1.000000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 110, 1.500000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 111, 1.750000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 112, 2.000000e+02, -2.500000e+01);
    err = jni_registerPointType(&env, eclass, 113, 2.000000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 114, 2.000000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 115, 2.000000e+02, 7.500000e+01);
    err = jni_registerPointType(&env, eclass, 116, 1.250000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 117, 1.500000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 118, 2.000000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 119, 2.250000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 120, 2.500000e+02, -2.500000e+01);
    err = jni_registerPointType(&env, eclass, 121, 2.500000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 122, 2.500000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 123, 2.500000e+02, 7.500000e+01);
    err = jni_registerPointType(&env, eclass, 124, 1.750000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 125, 2.000000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 126, 2.500000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 127, 2.750000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 128, 3.000000e+02, -2.500000e+01);
    err = jni_registerPointType(&env, eclass, 129, 3.000000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 130, 3.000000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 131, 3.000000e+02, 7.500000e+01);
    err = jni_registerPointType(&env, eclass, 132, 2.250000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 133, 2.500000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 134, 3.000000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 135, 3.250000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 136, 3.500000e+02, -2.500000e+01);
    err = jni_registerPointType(&env, eclass, 137, 3.500000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 138, 3.500000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 139, 3.500000e+02, 7.500000e+01);
    err = jni_registerPointType(&env, eclass, 140, 2.750000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 141, 3.000000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 142, 3.500000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 143, 3.750000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 144, 4.000000e+02, -2.500000e+01);
    err = jni_registerPointType(&env, eclass, 145, 4.000000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 146, 4.000000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 147, 4.000000e+02, 7.500000e+01);
    err = jni_registerPointType(&env, eclass, 148, 3.250000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 149, 3.500000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 150, 4.000000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 151, 4.250000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 152, 4.500000e+02, -2.500000e+01);
    err = jni_registerPointType(&env, eclass, 153, 4.500000e+02, 0.000000e+00);
    err = jni_registerPointType(&env, eclass, 154, 4.500000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 155, 4.500000e+02, 7.500000e+01);
    err = jni_registerPointType(&env, eclass, 156, 3.750000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 157, 4.000000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 158, 4.500000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 159, 4.750000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 160, 0.000000e+00, 2.500000e+01);
    err = jni_registerPointType(&env, eclass, 161, 0.000000e+00, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 162, 0.000000e+00, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 163, 0.000000e+00, 1.250000e+02);
    err = jni_registerPointType(&env, eclass, 164, -7.500000e+01, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 165, -5.000000e+01, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 166, 0.000000e+00, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 167, 2.500000e+01, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 168, 5.000000e+01, 2.500000e+01);
    err = jni_registerPointType(&env, eclass, 169, 5.000000e+01, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 170, 5.000000e+01, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 171, 5.000000e+01, 1.250000e+02);
    err = jni_registerPointType(&env, eclass, 172, -2.500000e+01, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 173, 0.000000e+00, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 174, 5.000000e+01, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 175, 7.500000e+01, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 176, 1.000000e+02, 2.500000e+01);
    err = jni_registerPointType(&env, eclass, 177, 1.000000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 178, 1.000000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 179, 1.000000e+02, 1.250000e+02);
    err = jni_registerPointType(&env, eclass, 180, 2.500000e+01, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 181, 5.000000e+01, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 182, 1.000000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 183, 1.250000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 184, 1.500000e+02, 2.500000e+01);
    err = jni_registerPointType(&env, eclass, 185, 1.500000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 186, 1.500000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 187, 1.500000e+02, 1.250000e+02);
    err = jni_registerPointType(&env, eclass, 188, 7.500000e+01, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 189, 1.000000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 190, 1.500000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 191, 1.750000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 192, 2.000000e+02, 2.500000e+01);
    err = jni_registerPointType(&env, eclass, 193, 2.000000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 194, 2.000000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 195, 2.000000e+02, 1.250000e+02);
    err = jni_registerPointType(&env, eclass, 196, 1.250000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 197, 1.500000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 198, 2.000000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 199, 2.250000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 200, 2.500000e+02, 2.500000e+01);
    err = jni_registerPointType(&env, eclass, 201, 2.500000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 202, 2.500000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 203, 2.500000e+02, 1.250000e+02);
    err = jni_registerPointType(&env, eclass, 204, 1.750000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 205, 2.000000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 206, 2.500000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 207, 2.750000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 208, 3.000000e+02, 2.500000e+01);
    err = jni_registerPointType(&env, eclass, 209, 3.000000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 210, 3.000000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 211, 3.000000e+02, 1.250000e+02);
    err = jni_registerPointType(&env, eclass, 212, 2.250000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 213, 2.500000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 214, 3.000000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 215, 3.250000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 216, 3.500000e+02, 2.500000e+01);
    err = jni_registerPointType(&env, eclass, 217, 3.500000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 218, 3.500000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 219, 3.500000e+02, 1.250000e+02);
    err = jni_registerPointType(&env, eclass, 220, 2.750000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 221, 3.000000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 222, 3.500000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 223, 3.750000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 224, 4.000000e+02, 2.500000e+01);
    err = jni_registerPointType(&env, eclass, 225, 4.000000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 226, 4.000000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 227, 4.000000e+02, 1.250000e+02);
    err = jni_registerPointType(&env, eclass, 228, 3.250000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 229, 3.500000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 230, 4.000000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 231, 4.250000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 232, 4.500000e+02, 2.500000e+01);
    err = jni_registerPointType(&env, eclass, 233, 4.500000e+02, 5.000000e+01);
    err = jni_registerPointType(&env, eclass, 234, 4.500000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 235, 4.500000e+02, 1.250000e+02);
    err = jni_registerPointType(&env, eclass, 236, 3.750000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 237, 4.000000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 238, 4.500000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 239, 4.750000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 240, 0.000000e+00, 7.500000e+01);
    err = jni_registerPointType(&env, eclass, 241, 0.000000e+00, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 242, 0.000000e+00, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 243, 0.000000e+00, 1.750000e+02);
    err = jni_registerPointType(&env, eclass, 244, -7.500000e+01, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 245, -5.000000e+01, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 246, 0.000000e+00, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 247, 2.500000e+01, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 248, 5.000000e+01, 7.500000e+01);
    err = jni_registerPointType(&env, eclass, 249, 5.000000e+01, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 250, 5.000000e+01, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 251, 5.000000e+01, 1.750000e+02);
    err = jni_registerPointType(&env, eclass, 252, -2.500000e+01, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 253, 0.000000e+00, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 254, 5.000000e+01, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 255, 7.500000e+01, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 256, 1.000000e+02, 7.500000e+01);
    err = jni_registerPointType(&env, eclass, 257, 1.000000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 258, 1.000000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 259, 1.000000e+02, 1.750000e+02);
    err = jni_registerPointType(&env, eclass, 260, 2.500000e+01, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 261, 5.000000e+01, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 262, 1.000000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 263, 1.250000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 264, 1.500000e+02, 7.500000e+01);
    err = jni_registerPointType(&env, eclass, 265, 1.500000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 266, 1.500000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 267, 1.500000e+02, 1.750000e+02);
    err = jni_registerPointType(&env, eclass, 268, 7.500000e+01, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 269, 1.000000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 270, 1.500000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 271, 1.750000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 272, 2.000000e+02, 7.500000e+01);
    err = jni_registerPointType(&env, eclass, 273, 2.000000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 274, 2.000000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 275, 2.000000e+02, 1.750000e+02);
    err = jni_registerPointType(&env, eclass, 276, 1.250000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 277, 1.500000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 278, 2.000000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 279, 2.250000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 280, 2.500000e+02, 7.500000e+01);
    err = jni_registerPointType(&env, eclass, 281, 2.500000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 282, 2.500000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 283, 2.500000e+02, 1.750000e+02);
    err = jni_registerPointType(&env, eclass, 284, 1.750000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 285, 2.000000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 286, 2.500000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 287, 2.750000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 288, 3.000000e+02, 7.500000e+01);
    err = jni_registerPointType(&env, eclass, 289, 3.000000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 290, 3.000000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 291, 3.000000e+02, 1.750000e+02);
    err = jni_registerPointType(&env, eclass, 292, 2.250000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 293, 2.500000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 294, 3.000000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 295, 3.250000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 296, 3.500000e+02, 7.500000e+01);
    err = jni_registerPointType(&env, eclass, 297, 3.500000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 298, 3.500000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 299, 3.500000e+02, 1.750000e+02);
    err = jni_registerPointType(&env, eclass, 300, 2.750000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 301, 3.000000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 302, 3.500000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 303, 3.750000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 304, 4.000000e+02, 7.500000e+01);
    err = jni_registerPointType(&env, eclass, 305, 4.000000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 306, 4.000000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 307, 4.000000e+02, 1.750000e+02);
    err = jni_registerPointType(&env, eclass, 308, 3.250000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 309, 3.500000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 310, 4.000000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 311, 4.250000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 312, 4.500000e+02, 7.500000e+01);
    err = jni_registerPointType(&env, eclass, 313, 4.500000e+02, 1.000000e+02);
    err = jni_registerPointType(&env, eclass, 314, 4.500000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 315, 4.500000e+02, 1.750000e+02);
    err = jni_registerPointType(&env, eclass, 316, 3.750000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 317, 4.000000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 318, 4.500000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 319, 4.750000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 320, 0.000000e+00, 1.250000e+02);
    err = jni_registerPointType(&env, eclass, 321, 0.000000e+00, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 322, 0.000000e+00, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 323, 0.000000e+00, 2.250000e+02);
    err = jni_registerPointType(&env, eclass, 324, -7.500000e+01, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 325, -5.000000e+01, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 326, 0.000000e+00, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 327, 2.500000e+01, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 328, 5.000000e+01, 1.250000e+02);
    err = jni_registerPointType(&env, eclass, 329, 5.000000e+01, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 330, 5.000000e+01, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 331, 5.000000e+01, 2.250000e+02);
    err = jni_registerPointType(&env, eclass, 332, -2.500000e+01, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 333, 0.000000e+00, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 334, 5.000000e+01, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 335, 7.500000e+01, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 336, 1.000000e+02, 1.250000e+02);
    err = jni_registerPointType(&env, eclass, 337, 1.000000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 338, 1.000000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 339, 1.000000e+02, 2.250000e+02);
    err = jni_registerPointType(&env, eclass, 340, 2.500000e+01, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 341, 5.000000e+01, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 342, 1.000000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 343, 1.250000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 344, 1.500000e+02, 1.250000e+02);
    err = jni_registerPointType(&env, eclass, 345, 1.500000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 346, 1.500000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 347, 1.500000e+02, 2.250000e+02);
    err = jni_registerPointType(&env, eclass, 348, 7.500000e+01, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 349, 1.000000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 350, 1.500000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 351, 1.750000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 352, 2.000000e+02, 1.250000e+02);
    err = jni_registerPointType(&env, eclass, 353, 2.000000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 354, 2.000000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 355, 2.000000e+02, 2.250000e+02);
    err = jni_registerPointType(&env, eclass, 356, 1.250000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 357, 1.500000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 358, 2.000000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 359, 2.250000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 360, 2.500000e+02, 1.250000e+02);
    err = jni_registerPointType(&env, eclass, 361, 2.500000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 362, 2.500000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 363, 2.500000e+02, 2.250000e+02);
    err = jni_registerPointType(&env, eclass, 364, 1.750000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 365, 2.000000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 366, 2.500000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 367, 2.750000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 368, 3.000000e+02, 1.250000e+02);
    err = jni_registerPointType(&env, eclass, 369, 3.000000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 370, 3.000000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 371, 3.000000e+02, 2.250000e+02);
    err = jni_registerPointType(&env, eclass, 372, 2.250000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 373, 2.500000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 374, 3.000000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 375, 3.250000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 376, 3.500000e+02, 1.250000e+02);
    err = jni_registerPointType(&env, eclass, 377, 3.500000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 378, 3.500000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 379, 3.500000e+02, 2.250000e+02);
    err = jni_registerPointType(&env, eclass, 380, 2.750000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 381, 3.000000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 382, 3.500000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 383, 3.750000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 384, 4.000000e+02, 1.250000e+02);
    err = jni_registerPointType(&env, eclass, 385, 4.000000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 386, 4.000000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 387, 4.000000e+02, 2.250000e+02);
    err = jni_registerPointType(&env, eclass, 388, 3.250000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 389, 3.500000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 390, 4.000000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 391, 4.250000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 392, 4.500000e+02, 1.250000e+02);
    err = jni_registerPointType(&env, eclass, 393, 4.500000e+02, 1.500000e+02);
    err = jni_registerPointType(&env, eclass, 394, 4.500000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 395, 4.500000e+02, 2.250000e+02);
    err = jni_registerPointType(&env, eclass, 396, 3.750000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 397, 4.000000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 398, 4.500000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 399, 4.750000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 400, 0.000000e+00, 1.750000e+02);
    err = jni_registerPointType(&env, eclass, 401, 0.000000e+00, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 402, 0.000000e+00, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 403, 0.000000e+00, 2.750000e+02);
    err = jni_registerPointType(&env, eclass, 404, -7.500000e+01, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 405, -5.000000e+01, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 406, 0.000000e+00, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 407, 2.500000e+01, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 408, 5.000000e+01, 1.750000e+02);
    err = jni_registerPointType(&env, eclass, 409, 5.000000e+01, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 410, 5.000000e+01, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 411, 5.000000e+01, 2.750000e+02);
    err = jni_registerPointType(&env, eclass, 412, -2.500000e+01, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 413, 0.000000e+00, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 414, 5.000000e+01, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 415, 7.500000e+01, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 416, 1.000000e+02, 1.750000e+02);
    err = jni_registerPointType(&env, eclass, 417, 1.000000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 418, 1.000000e+02, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 419, 1.000000e+02, 2.750000e+02);
    err = jni_registerPointType(&env, eclass, 420, 2.500000e+01, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 421, 5.000000e+01, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 422, 1.000000e+02, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 423, 1.250000e+02, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 424, 1.500000e+02, 1.750000e+02);
    err = jni_registerPointType(&env, eclass, 425, 1.500000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 426, 1.500000e+02, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 427, 1.500000e+02, 2.750000e+02);
    err = jni_registerPointType(&env, eclass, 428, 7.500000e+01, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 429, 1.000000e+02, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 430, 1.500000e+02, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 431, 1.750000e+02, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 432, 2.000000e+02, 1.750000e+02);
    err = jni_registerPointType(&env, eclass, 433, 2.000000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 434, 2.000000e+02, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 435, 2.000000e+02, 2.750000e+02);
    err = jni_registerPointType(&env, eclass, 436, 1.250000e+02, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 437, 1.500000e+02, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 438, 2.000000e+02, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 439, 2.250000e+02, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 440, 2.500000e+02, 1.750000e+02);
    err = jni_registerPointType(&env, eclass, 441, 2.500000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 442, 2.500000e+02, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 443, 2.500000e+02, 2.750000e+02);
    err = jni_registerPointType(&env, eclass, 444, 1.750000e+02, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 445, 2.000000e+02, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 446, 2.500000e+02, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 447, 2.750000e+02, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 448, 3.000000e+02, 1.750000e+02);
    err = jni_registerPointType(&env, eclass, 449, 3.000000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 450, 3.000000e+02, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 451, 3.000000e+02, 2.750000e+02);
    err = jni_registerPointType(&env, eclass, 452, 2.250000e+02, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 453, 2.500000e+02, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 454, 3.000000e+02, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 455, 3.250000e+02, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 456, 3.500000e+02, 1.750000e+02);
    err = jni_registerPointType(&env, eclass, 457, 3.500000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 458, 3.500000e+02, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 459, 3.500000e+02, 2.750000e+02);
    err = jni_registerPointType(&env, eclass, 460, 2.750000e+02, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 461, 3.000000e+02, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 462, 3.500000e+02, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 463, 3.750000e+02, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 464, 4.000000e+02, 1.750000e+02);
    err = jni_registerPointType(&env, eclass, 465, 4.000000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 466, 4.000000e+02, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 467, 4.000000e+02, 2.750000e+02);
    err = jni_registerPointType(&env, eclass, 468, 3.250000e+02, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 469, 3.500000e+02, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 470, 4.000000e+02, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 471, 4.250000e+02, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 472, 4.500000e+02, 1.750000e+02);
    err = jni_registerPointType(&env, eclass, 473, 4.500000e+02, 2.000000e+02);
    err = jni_registerPointType(&env, eclass, 474, 4.500000e+02, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 475, 4.500000e+02, 2.750000e+02);
    err = jni_registerPointType(&env, eclass, 476, 3.750000e+02, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 477, 4.000000e+02, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 478, 4.500000e+02, 2.500000e+02);
    err = jni_registerPointType(&env, eclass, 479, 4.750000e+02, 2.500000e+02);

    err = jni_registerGeometricType(&env, eclass, 1, GEOMETRIC_TYPE_ID_LINE, 1, 2, -1, 0, 3, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 2, GEOMETRIC_TYPE_ID_LINE, 5, 6, -1, 4, 7, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 3, GEOMETRIC_TYPE_ID_LINE, 9, 10, -1, 8, 11, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 4, GEOMETRIC_TYPE_ID_LINE, 13, 14, -1, 12, 15, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 5, GEOMETRIC_TYPE_ID_LINE, 17, 18, -1, 16, 19, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 6, GEOMETRIC_TYPE_ID_LINE, 21, 22, -1, 20, 23, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 7, GEOMETRIC_TYPE_ID_LINE, 25, 26, -1, 24, 27, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 8, GEOMETRIC_TYPE_ID_LINE, 29, 30, -1, 28, 31, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 9, GEOMETRIC_TYPE_ID_LINE, 33, 34, -1, 32, 35, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 10, GEOMETRIC_TYPE_ID_LINE, 37, 38, -1, 36, 39, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 11, GEOMETRIC_TYPE_ID_LINE, 41, 42, -1, 40, 43, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 12, GEOMETRIC_TYPE_ID_LINE, 45, 46, -1, 44, 47, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 13, GEOMETRIC_TYPE_ID_LINE, 49, 50, -1, 48, 51, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 14, GEOMETRIC_TYPE_ID_LINE, 53, 54, -1, 52, 55, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 15, GEOMETRIC_TYPE_ID_LINE, 57, 58, -1, 56, 59, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 16, GEOMETRIC_TYPE_ID_LINE, 61, 62, -1, 60, 63, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 17, GEOMETRIC_TYPE_ID_LINE, 65, 66, -1, 64, 67, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 18, GEOMETRIC_TYPE_ID_LINE, 69, 70, -1, 68, 71, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 19, GEOMETRIC_TYPE_ID_LINE, 73, 74, -1, 72, 75, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 20, GEOMETRIC_TYPE_ID_LINE, 77, 78, -1, 76, 79, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 21, GEOMETRIC_TYPE_ID_LINE, 81, 82, -1, 80, 83, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 22, GEOMETRIC_TYPE_ID_LINE, 85, 86, -1, 84, 87, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 23, GEOMETRIC_TYPE_ID_LINE, 89, 90, -1, 88, 91, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 24, GEOMETRIC_TYPE_ID_LINE, 93, 94, -1, 92, 95, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 25, GEOMETRIC_TYPE_ID_LINE, 97, 98, -1, 96, 99, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 26, GEOMETRIC_TYPE_ID_LINE, 101, 102, -1, 100, 103, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 27, GEOMETRIC_TYPE_ID_LINE, 105, 106, -1, 104, 107, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 28, GEOMETRIC_TYPE_ID_LINE, 109, 110, -1, 108, 111, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 29, GEOMETRIC_TYPE_ID_LINE, 113, 114, -1, 112, 115, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 30, GEOMETRIC_TYPE_ID_LINE, 117, 118, -1, 116, 119, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 31, GEOMETRIC_TYPE_ID_LINE, 121, 122, -1, 120, 123, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 32, GEOMETRIC_TYPE_ID_LINE, 125, 126, -1, 124, 127, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 33, GEOMETRIC_TYPE_ID_LINE, 129, 130, -1, 128, 131, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 34, GEOMETRIC_TYPE_ID_LINE, 133, 134, -1, 132, 135, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 35, GEOMETRIC_TYPE_ID_LINE, 137, 138, -1, 136, 139, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 36, GEOMETRIC_TYPE_ID_LINE, 141, 142, -1, 140, 143, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 37, GEOMETRIC_TYPE_ID_LINE, 145, 146, -1, 144, 147, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 38, GEOMETRIC_TYPE_ID_LINE, 149, 150, -1, 148, 151, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 39, GEOMETRIC_TYPE_ID_LINE, 153, 154, -1, 152, 155, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 40, GEOMETRIC_TYPE_ID_LINE, 157, 158, -1, 156, 159, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 41, GEOMETRIC_TYPE_ID_LINE, 161, 162, -1, 160, 163, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 42, GEOMETRIC_TYPE_ID_LINE, 165, 166, -1, 164, 167, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 43, GEOMETRIC_TYPE_ID_LINE, 169, 170, -1, 168, 171, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 44, GEOMETRIC_TYPE_ID_LINE, 173, 174, -1, 172, 175, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 45, GEOMETRIC_TYPE_ID_LINE, 177, 178, -1, 176, 179, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 46, GEOMETRIC_TYPE_ID_LINE, 181, 182, -1, 180, 183, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 47, GEOMETRIC_TYPE_ID_LINE, 185, 186, -1, 184, 187, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 48, GEOMETRIC_TYPE_ID_LINE, 189, 190, -1, 188, 191, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 49, GEOMETRIC_TYPE_ID_LINE, 193, 194, -1, 192, 195, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 50, GEOMETRIC_TYPE_ID_LINE, 197, 198, -1, 196, 199, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 51, GEOMETRIC_TYPE_ID_LINE, 201, 202, -1, 200, 203, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 52, GEOMETRIC_TYPE_ID_LINE, 205, 206, -1, 204, 207, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 53, GEOMETRIC_TYPE_ID_LINE, 209, 210, -1, 208, 211, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 54, GEOMETRIC_TYPE_ID_LINE, 213, 214, -1, 212, 215, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 55, GEOMETRIC_TYPE_ID_LINE, 217, 218, -1, 216, 219, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 56, GEOMETRIC_TYPE_ID_LINE, 221, 222, -1, 220, 223, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 57, GEOMETRIC_TYPE_ID_LINE, 225, 226, -1, 224, 227, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 58, GEOMETRIC_TYPE_ID_LINE, 229, 230, -1, 228, 231, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 59, GEOMETRIC_TYPE_ID_LINE, 233, 234, -1, 232, 235, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 60, GEOMETRIC_TYPE_ID_LINE, 237, 238, -1, 236, 239, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 61, GEOMETRIC_TYPE_ID_LINE, 241, 242, -1, 240, 243, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 62, GEOMETRIC_TYPE_ID_LINE, 245, 246, -1, 244, 247, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 63, GEOMETRIC_TYPE_ID_LINE, 249, 250, -1, 248, 251, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 64, GEOMETRIC_TYPE_ID_LINE, 253, 254, -1, 252, 255, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 65, GEOMETRIC_TYPE_ID_LINE, 257, 258, -1, 256, 259, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 66, GEOMETRIC_TYPE_ID_LINE, 261, 262, -1, 260, 263, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 67, GEOMETRIC_TYPE_ID_LINE, 265, 266, -1, 264, 267, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 68, GEOMETRIC_TYPE_ID_LINE, 269, 270, -1, 268, 271, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 69, GEOMETRIC_TYPE_ID_LINE, 273, 274, -1, 272, 275, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 70, GEOMETRIC_TYPE_ID_LINE, 277, 278, -1, 276, 279, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 71, GEOMETRIC_TYPE_ID_LINE, 281, 282, -1, 280, 283, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 72, GEOMETRIC_TYPE_ID_LINE, 285, 286, -1, 284, 287, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 73, GEOMETRIC_TYPE_ID_LINE, 289, 290, -1, 288, 291, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 74, GEOMETRIC_TYPE_ID_LINE, 293, 294, -1, 292, 295, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 75, GEOMETRIC_TYPE_ID_LINE, 297, 298, -1, 296, 299, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 76, GEOMETRIC_TYPE_ID_LINE, 301, 302, -1, 300, 303, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 77, GEOMETRIC_TYPE_ID_LINE, 305, 306, -1, 304, 307, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 78, GEOMETRIC_TYPE_ID_LINE, 309, 310, -1, 308, 311, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 79, GEOMETRIC_TYPE_ID_LINE, 313, 314, -1, 312, 315, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 80, GEOMETRIC_TYPE_ID_LINE, 317, 318, -1, 316, 319, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 81, GEOMETRIC_TYPE_ID_LINE, 321, 322, -1, 320, 323, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 82, GEOMETRIC_TYPE_ID_LINE, 325, 326, -1, 324, 327, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 83, GEOMETRIC_TYPE_ID_LINE, 329, 330, -1, 328, 331, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 84, GEOMETRIC_TYPE_ID_LINE, 333, 334, -1, 332, 335, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 85, GEOMETRIC_TYPE_ID_LINE, 337, 338, -1, 336, 339, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 86, GEOMETRIC_TYPE_ID_LINE, 341, 342, -1, 340, 343, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 87, GEOMETRIC_TYPE_ID_LINE, 345, 346, -1, 344, 347, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 88, GEOMETRIC_TYPE_ID_LINE, 349, 350, -1, 348, 351, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 89, GEOMETRIC_TYPE_ID_LINE, 353, 354, -1, 352, 355, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 90, GEOMETRIC_TYPE_ID_LINE, 357, 358, -1, 356, 359, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 91, GEOMETRIC_TYPE_ID_LINE, 361, 362, -1, 360, 363, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 92, GEOMETRIC_TYPE_ID_LINE, 365, 366, -1, 364, 367, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 93, GEOMETRIC_TYPE_ID_LINE, 369, 370, -1, 368, 371, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 94, GEOMETRIC_TYPE_ID_LINE, 373, 374, -1, 372, 375, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 95, GEOMETRIC_TYPE_ID_LINE, 377, 378, -1, 376, 379, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 96, GEOMETRIC_TYPE_ID_LINE, 381, 382, -1, 380, 383, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 97, GEOMETRIC_TYPE_ID_LINE, 385, 386, -1, 384, 387, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 98, GEOMETRIC_TYPE_ID_LINE, 389, 390, -1, 388, 391, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 99, GEOMETRIC_TYPE_ID_LINE, 393, 394, -1, 392, 395, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 100, GEOMETRIC_TYPE_ID_LINE, 397, 398, -1, 396, 399, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 101, GEOMETRIC_TYPE_ID_LINE, 401, 402, -1, 400, 403, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 102, GEOMETRIC_TYPE_ID_LINE, 405, 406, -1, 404, 407, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 103, GEOMETRIC_TYPE_ID_LINE, 409, 410, -1, 408, 411, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 104, GEOMETRIC_TYPE_ID_LINE, 413, 414, -1, 412, 415, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 105, GEOMETRIC_TYPE_ID_LINE, 417, 418, -1, 416, 419, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 106, GEOMETRIC_TYPE_ID_LINE, 421, 422, -1, 420, 423, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 107, GEOMETRIC_TYPE_ID_LINE, 425, 426, -1, 424, 427, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 108, GEOMETRIC_TYPE_ID_LINE, 429, 430, -1, 428, 431, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 109, GEOMETRIC_TYPE_ID_LINE, 433, 434, -1, 432, 435, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 110, GEOMETRIC_TYPE_ID_LINE, 437, 438, -1, 436, 439, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 111, GEOMETRIC_TYPE_ID_LINE, 441, 442, -1, 440, 443, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 112, GEOMETRIC_TYPE_ID_LINE, 445, 446, -1, 444, 447, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 113, GEOMETRIC_TYPE_ID_LINE, 449, 450, -1, 448, 451, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 114, GEOMETRIC_TYPE_ID_LINE, 453, 454, -1, 452, 455, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 115, GEOMETRIC_TYPE_ID_LINE, 457, 458, -1, 456, 459, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 116, GEOMETRIC_TYPE_ID_LINE, 461, 462, -1, 460, 463, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 117, GEOMETRIC_TYPE_ID_LINE, 465, 466, -1, 464, 467, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 118, GEOMETRIC_TYPE_ID_LINE, 469, 470, -1, 468, 471, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 119, GEOMETRIC_TYPE_ID_LINE, 473, 474, -1, 472, 475, -1, -1);
    err = jni_registerGeometricType(&env, eclass, 120, GEOMETRIC_TYPE_ID_LINE, 477, 478, -1, 476, 479, -1, -1);

    err = jni_registerConstraintType(&env, eclass, 0, CONSTRAINT_TYPE_ID_FIX_POINT, 0, -1, -1, -1, -1, 0.000000e+00, -7.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 1, CONSTRAINT_TYPE_ID_FIX_POINT, 3, -1, -1, -1, -1, 0.000000e+00, 2.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 2, CONSTRAINT_TYPE_ID_FIX_POINT, 4, -1, -1, -1, -1, -7.500000e+01, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 3, CONSTRAINT_TYPE_ID_FIX_POINT, 7, -1, -1, -1, -1, 2.500000e+01, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 4, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 6, 2, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 5, CONSTRAINT_TYPE_ID_FIX_POINT, 8, -1, -1, -1, -1, 5.000000e+01, -7.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 6, CONSTRAINT_TYPE_ID_FIX_POINT, 11, -1, -1, -1, -1, 5.000000e+01, 2.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 7, CONSTRAINT_TYPE_ID_FIX_POINT, 12, -1, -1, -1, -1, -2.500000e+01, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 8, CONSTRAINT_TYPE_ID_FIX_POINT, 15, -1, -1, -1, -1, 7.500000e+01, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 9, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 14, 10, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 10, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 13, 6, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 11, CONSTRAINT_TYPE_ID_FIX_POINT, 16, -1, -1, -1, -1, 1.000000e+02, -7.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 12, CONSTRAINT_TYPE_ID_FIX_POINT, 19, -1, -1, -1, -1, 1.000000e+02, 2.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 13, CONSTRAINT_TYPE_ID_FIX_POINT, 20, -1, -1, -1, -1, 2.500000e+01, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 14, CONSTRAINT_TYPE_ID_FIX_POINT, 23, -1, -1, -1, -1, 1.250000e+02, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 15, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 22, 18, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 16, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 21, 14, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 17, CONSTRAINT_TYPE_ID_FIX_POINT, 24, -1, -1, -1, -1, 1.500000e+02, -7.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 18, CONSTRAINT_TYPE_ID_FIX_POINT, 27, -1, -1, -1, -1, 1.500000e+02, 2.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 19, CONSTRAINT_TYPE_ID_FIX_POINT, 28, -1, -1, -1, -1, 7.500000e+01, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 20, CONSTRAINT_TYPE_ID_FIX_POINT, 31, -1, -1, -1, -1, 1.750000e+02, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 21, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 30, 26, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 22, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 29, 22, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 23, CONSTRAINT_TYPE_ID_FIX_POINT, 32, -1, -1, -1, -1, 2.000000e+02, -7.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 24, CONSTRAINT_TYPE_ID_FIX_POINT, 35, -1, -1, -1, -1, 2.000000e+02, 2.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 25, CONSTRAINT_TYPE_ID_FIX_POINT, 36, -1, -1, -1, -1, 1.250000e+02, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 26, CONSTRAINT_TYPE_ID_FIX_POINT, 39, -1, -1, -1, -1, 2.250000e+02, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 27, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 38, 34, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 28, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 37, 30, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 29, CONSTRAINT_TYPE_ID_FIX_POINT, 40, -1, -1, -1, -1, 2.500000e+02, -7.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 30, CONSTRAINT_TYPE_ID_FIX_POINT, 43, -1, -1, -1, -1, 2.500000e+02, 2.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 31, CONSTRAINT_TYPE_ID_FIX_POINT, 44, -1, -1, -1, -1, 1.750000e+02, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 32, CONSTRAINT_TYPE_ID_FIX_POINT, 47, -1, -1, -1, -1, 2.750000e+02, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 33, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 46, 42, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 34, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 45, 38, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 35, CONSTRAINT_TYPE_ID_FIX_POINT, 48, -1, -1, -1, -1, 3.000000e+02, -7.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 36, CONSTRAINT_TYPE_ID_FIX_POINT, 51, -1, -1, -1, -1, 3.000000e+02, 2.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 37, CONSTRAINT_TYPE_ID_FIX_POINT, 52, -1, -1, -1, -1, 2.250000e+02, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 38, CONSTRAINT_TYPE_ID_FIX_POINT, 55, -1, -1, -1, -1, 3.250000e+02, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 39, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 54, 50, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 40, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 53, 46, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 41, CONSTRAINT_TYPE_ID_FIX_POINT, 56, -1, -1, -1, -1, 3.500000e+02, -7.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 42, CONSTRAINT_TYPE_ID_FIX_POINT, 59, -1, -1, -1, -1, 3.500000e+02, 2.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 43, CONSTRAINT_TYPE_ID_FIX_POINT, 60, -1, -1, -1, -1, 2.750000e+02, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 44, CONSTRAINT_TYPE_ID_FIX_POINT, 63, -1, -1, -1, -1, 3.750000e+02, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 45, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 62, 58, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 46, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 61, 54, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 47, CONSTRAINT_TYPE_ID_FIX_POINT, 64, -1, -1, -1, -1, 4.000000e+02, -7.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 48, CONSTRAINT_TYPE_ID_FIX_POINT, 67, -1, -1, -1, -1, 4.000000e+02, 2.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 49, CONSTRAINT_TYPE_ID_FIX_POINT, 68, -1, -1, -1, -1, 3.250000e+02, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 50, CONSTRAINT_TYPE_ID_FIX_POINT, 71, -1, -1, -1, -1, 4.250000e+02, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 51, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 70, 66, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 52, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 69, 62, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 53, CONSTRAINT_TYPE_ID_FIX_POINT, 72, -1, -1, -1, -1, 4.500000e+02, -7.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 54, CONSTRAINT_TYPE_ID_FIX_POINT, 75, -1, -1, -1, -1, 4.500000e+02, 2.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 55, CONSTRAINT_TYPE_ID_FIX_POINT, 76, -1, -1, -1, -1, 3.750000e+02, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 56, CONSTRAINT_TYPE_ID_FIX_POINT, 79, -1, -1, -1, -1, 4.750000e+02, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 57, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 78, 74, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 58, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 77, 70, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 59, CONSTRAINT_TYPE_ID_FIX_POINT, 80, -1, -1, -1, -1, 0.000000e+00, -2.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 60, CONSTRAINT_TYPE_ID_FIX_POINT, 83, -1, -1, -1, -1, 0.000000e+00, 7.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 61, CONSTRAINT_TYPE_ID_FIX_POINT, 84, -1, -1, -1, -1, -7.500000e+01, 5.000000e+01);
    err = jni_registerConstraintType(&env, eclass, 62, CONSTRAINT_TYPE_ID_FIX_POINT, 87, -1, -1, -1, -1, 2.500000e+01, 5.000000e+01);
    err = jni_registerConstraintType(&env, eclass, 63, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 86, 82, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 64, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 81, 2, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 65, CONSTRAINT_TYPE_ID_FIX_POINT, 88, -1, -1, -1, -1, 5.000000e+01, -2.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 66, CONSTRAINT_TYPE_ID_FIX_POINT, 91, -1, -1, -1, -1, 5.000000e+01, 7.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 67, CONSTRAINT_TYPE_ID_FIX_POINT, 92, -1, -1, -1, -1, -2.500000e+01, 5.000000e+01);
    err = jni_registerConstraintType(&env, eclass, 68, CONSTRAINT_TYPE_ID_FIX_POINT, 95, -1, -1, -1, -1, 7.500000e+01, 5.000000e+01);
    err = jni_registerConstraintType(&env, eclass, 69, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 94, 90, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 70, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 89, 10, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 71, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 93, 86, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 72, CONSTRAINT_TYPE_ID_FIX_POINT, 96, -1, -1, -1, -1, 1.000000e+02, -2.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 73, CONSTRAINT_TYPE_ID_FIX_POINT, 99, -1, -1, -1, -1, 1.000000e+02, 7.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 74, CONSTRAINT_TYPE_ID_FIX_POINT, 100, -1, -1, -1, -1, 2.500000e+01, 5.000000e+01);
    err = jni_registerConstraintType(&env, eclass, 75, CONSTRAINT_TYPE_ID_FIX_POINT, 103, -1, -1, -1, -1, 1.250000e+02, 5.000000e+01);
    err = jni_registerConstraintType(&env, eclass, 76, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 102, 98, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 77, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 97, 18, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 78, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 101, 94, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 79, CONSTRAINT_TYPE_ID_FIX_POINT, 104, -1, -1, -1, -1, 1.500000e+02, -2.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 80, CONSTRAINT_TYPE_ID_FIX_POINT, 107, -1, -1, -1, -1, 1.500000e+02, 7.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 81, CONSTRAINT_TYPE_ID_FIX_POINT, 108, -1, -1, -1, -1, 7.500000e+01, 5.000000e+01);
    err = jni_registerConstraintType(&env, eclass, 82, CONSTRAINT_TYPE_ID_FIX_POINT, 111, -1, -1, -1, -1, 1.750000e+02, 5.000000e+01);
    err = jni_registerConstraintType(&env, eclass, 83, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 110, 106, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 84, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 105, 26, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 85, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 109, 102, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 86, CONSTRAINT_TYPE_ID_FIX_POINT, 112, -1, -1, -1, -1, 2.000000e+02, -2.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 87, CONSTRAINT_TYPE_ID_FIX_POINT, 115, -1, -1, -1, -1, 2.000000e+02, 7.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 88, CONSTRAINT_TYPE_ID_FIX_POINT, 116, -1, -1, -1, -1, 1.250000e+02, 5.000000e+01);
    err = jni_registerConstraintType(&env, eclass, 89, CONSTRAINT_TYPE_ID_FIX_POINT, 119, -1, -1, -1, -1, 2.250000e+02, 5.000000e+01);
    err = jni_registerConstraintType(&env, eclass, 90, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 118, 114, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 91, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 113, 34, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 92, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 117, 110, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 93, CONSTRAINT_TYPE_ID_FIX_POINT, 120, -1, -1, -1, -1, 2.500000e+02, -2.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 94, CONSTRAINT_TYPE_ID_FIX_POINT, 123, -1, -1, -1, -1, 2.500000e+02, 7.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 95, CONSTRAINT_TYPE_ID_FIX_POINT, 124, -1, -1, -1, -1, 1.750000e+02, 5.000000e+01);
    err = jni_registerConstraintType(&env, eclass, 96, CONSTRAINT_TYPE_ID_FIX_POINT, 127, -1, -1, -1, -1, 2.750000e+02, 5.000000e+01);
    err = jni_registerConstraintType(&env, eclass, 97, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 126, 122, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 98, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 121, 42, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 99, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 125, 118, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 100, CONSTRAINT_TYPE_ID_FIX_POINT, 128, -1, -1, -1, -1, 3.000000e+02, -2.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 101, CONSTRAINT_TYPE_ID_FIX_POINT, 131, -1, -1, -1, -1, 3.000000e+02, 7.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 102, CONSTRAINT_TYPE_ID_FIX_POINT, 132, -1, -1, -1, -1, 2.250000e+02, 5.000000e+01);
    err = jni_registerConstraintType(&env, eclass, 103, CONSTRAINT_TYPE_ID_FIX_POINT, 135, -1, -1, -1, -1, 3.250000e+02, 5.000000e+01);
    err = jni_registerConstraintType(&env, eclass, 104, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 134, 130, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 105, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 129, 50, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 106, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 133, 126, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 107, CONSTRAINT_TYPE_ID_FIX_POINT, 136, -1, -1, -1, -1, 3.500000e+02, -2.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 108, CONSTRAINT_TYPE_ID_FIX_POINT, 139, -1, -1, -1, -1, 3.500000e+02, 7.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 109, CONSTRAINT_TYPE_ID_FIX_POINT, 140, -1, -1, -1, -1, 2.750000e+02, 5.000000e+01);
    err = jni_registerConstraintType(&env, eclass, 110, CONSTRAINT_TYPE_ID_FIX_POINT, 143, -1, -1, -1, -1, 3.750000e+02, 5.000000e+01);
    err = jni_registerConstraintType(&env, eclass, 111, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 142, 138, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 112, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 137, 58, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 113, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 141, 134, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 114, CONSTRAINT_TYPE_ID_FIX_POINT, 144, -1, -1, -1, -1, 4.000000e+02, -2.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 115, CONSTRAINT_TYPE_ID_FIX_POINT, 147, -1, -1, -1, -1, 4.000000e+02, 7.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 116, CONSTRAINT_TYPE_ID_FIX_POINT, 148, -1, -1, -1, -1, 3.250000e+02, 5.000000e+01);
    err = jni_registerConstraintType(&env, eclass, 117, CONSTRAINT_TYPE_ID_FIX_POINT, 151, -1, -1, -1, -1, 4.250000e+02, 5.000000e+01);
    err = jni_registerConstraintType(&env, eclass, 118, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 150, 146, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 119, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 145, 66, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 120, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 149, 142, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 121, CONSTRAINT_TYPE_ID_FIX_POINT, 152, -1, -1, -1, -1, 4.500000e+02, -2.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 122, CONSTRAINT_TYPE_ID_FIX_POINT, 155, -1, -1, -1, -1, 4.500000e+02, 7.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 123, CONSTRAINT_TYPE_ID_FIX_POINT, 156, -1, -1, -1, -1, 3.750000e+02, 5.000000e+01);
    err = jni_registerConstraintType(&env, eclass, 124, CONSTRAINT_TYPE_ID_FIX_POINT, 159, -1, -1, -1, -1, 4.750000e+02, 5.000000e+01);
    err = jni_registerConstraintType(&env, eclass, 125, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 158, 154, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 126, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 153, 74, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 127, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 157, 150, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 128, CONSTRAINT_TYPE_ID_FIX_POINT, 160, -1, -1, -1, -1, 0.000000e+00, 2.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 129, CONSTRAINT_TYPE_ID_FIX_POINT, 163, -1, -1, -1, -1, 0.000000e+00, 1.250000e+02);
    err = jni_registerConstraintType(&env, eclass, 130, CONSTRAINT_TYPE_ID_FIX_POINT, 164, -1, -1, -1, -1, -7.500000e+01, 1.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 131, CONSTRAINT_TYPE_ID_FIX_POINT, 167, -1, -1, -1, -1, 2.500000e+01, 1.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 132, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 166, 162, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 133, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 161, 82, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 134, CONSTRAINT_TYPE_ID_FIX_POINT, 168, -1, -1, -1, -1, 5.000000e+01, 2.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 135, CONSTRAINT_TYPE_ID_FIX_POINT, 171, -1, -1, -1, -1, 5.000000e+01, 1.250000e+02);
    err = jni_registerConstraintType(&env, eclass, 136, CONSTRAINT_TYPE_ID_FIX_POINT, 172, -1, -1, -1, -1, -2.500000e+01, 1.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 137, CONSTRAINT_TYPE_ID_FIX_POINT, 175, -1, -1, -1, -1, 7.500000e+01, 1.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 138, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 174, 170, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 139, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 169, 90, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 140, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 173, 166, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 141, CONSTRAINT_TYPE_ID_FIX_POINT, 176, -1, -1, -1, -1, 1.000000e+02, 2.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 142, CONSTRAINT_TYPE_ID_FIX_POINT, 179, -1, -1, -1, -1, 1.000000e+02, 1.250000e+02);
    err = jni_registerConstraintType(&env, eclass, 143, CONSTRAINT_TYPE_ID_FIX_POINT, 180, -1, -1, -1, -1, 2.500000e+01, 1.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 144, CONSTRAINT_TYPE_ID_FIX_POINT, 183, -1, -1, -1, -1, 1.250000e+02, 1.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 145, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 182, 178, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 146, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 177, 98, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 147, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 181, 174, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 148, CONSTRAINT_TYPE_ID_FIX_POINT, 184, -1, -1, -1, -1, 1.500000e+02, 2.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 149, CONSTRAINT_TYPE_ID_FIX_POINT, 187, -1, -1, -1, -1, 1.500000e+02, 1.250000e+02);
    err = jni_registerConstraintType(&env, eclass, 150, CONSTRAINT_TYPE_ID_FIX_POINT, 188, -1, -1, -1, -1, 7.500000e+01, 1.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 151, CONSTRAINT_TYPE_ID_FIX_POINT, 191, -1, -1, -1, -1, 1.750000e+02, 1.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 152, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 190, 186, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 153, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 185, 106, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 154, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 189, 182, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 155, CONSTRAINT_TYPE_ID_FIX_POINT, 192, -1, -1, -1, -1, 2.000000e+02, 2.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 156, CONSTRAINT_TYPE_ID_FIX_POINT, 195, -1, -1, -1, -1, 2.000000e+02, 1.250000e+02);
    err = jni_registerConstraintType(&env, eclass, 157, CONSTRAINT_TYPE_ID_FIX_POINT, 196, -1, -1, -1, -1, 1.250000e+02, 1.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 158, CONSTRAINT_TYPE_ID_FIX_POINT, 199, -1, -1, -1, -1, 2.250000e+02, 1.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 159, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 198, 194, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 160, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 193, 114, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 161, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 197, 190, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 162, CONSTRAINT_TYPE_ID_FIX_POINT, 200, -1, -1, -1, -1, 2.500000e+02, 2.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 163, CONSTRAINT_TYPE_ID_FIX_POINT, 203, -1, -1, -1, -1, 2.500000e+02, 1.250000e+02);
    err = jni_registerConstraintType(&env, eclass, 164, CONSTRAINT_TYPE_ID_FIX_POINT, 204, -1, -1, -1, -1, 1.750000e+02, 1.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 165, CONSTRAINT_TYPE_ID_FIX_POINT, 207, -1, -1, -1, -1, 2.750000e+02, 1.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 166, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 206, 202, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 167, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 201, 122, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 168, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 205, 198, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 169, CONSTRAINT_TYPE_ID_FIX_POINT, 208, -1, -1, -1, -1, 3.000000e+02, 2.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 170, CONSTRAINT_TYPE_ID_FIX_POINT, 211, -1, -1, -1, -1, 3.000000e+02, 1.250000e+02);
    err = jni_registerConstraintType(&env, eclass, 171, CONSTRAINT_TYPE_ID_FIX_POINT, 212, -1, -1, -1, -1, 2.250000e+02, 1.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 172, CONSTRAINT_TYPE_ID_FIX_POINT, 215, -1, -1, -1, -1, 3.250000e+02, 1.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 173, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 214, 210, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 174, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 209, 130, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 175, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 213, 206, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 176, CONSTRAINT_TYPE_ID_FIX_POINT, 216, -1, -1, -1, -1, 3.500000e+02, 2.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 177, CONSTRAINT_TYPE_ID_FIX_POINT, 219, -1, -1, -1, -1, 3.500000e+02, 1.250000e+02);
    err = jni_registerConstraintType(&env, eclass, 178, CONSTRAINT_TYPE_ID_FIX_POINT, 220, -1, -1, -1, -1, 2.750000e+02, 1.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 179, CONSTRAINT_TYPE_ID_FIX_POINT, 223, -1, -1, -1, -1, 3.750000e+02, 1.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 180, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 222, 218, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 181, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 217, 138, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 182, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 221, 214, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 183, CONSTRAINT_TYPE_ID_FIX_POINT, 224, -1, -1, -1, -1, 4.000000e+02, 2.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 184, CONSTRAINT_TYPE_ID_FIX_POINT, 227, -1, -1, -1, -1, 4.000000e+02, 1.250000e+02);
    err = jni_registerConstraintType(&env, eclass, 185, CONSTRAINT_TYPE_ID_FIX_POINT, 228, -1, -1, -1, -1, 3.250000e+02, 1.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 186, CONSTRAINT_TYPE_ID_FIX_POINT, 231, -1, -1, -1, -1, 4.250000e+02, 1.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 187, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 230, 226, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 188, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 225, 146, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 189, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 229, 222, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 190, CONSTRAINT_TYPE_ID_FIX_POINT, 232, -1, -1, -1, -1, 4.500000e+02, 2.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 191, CONSTRAINT_TYPE_ID_FIX_POINT, 235, -1, -1, -1, -1, 4.500000e+02, 1.250000e+02);
    err = jni_registerConstraintType(&env, eclass, 192, CONSTRAINT_TYPE_ID_FIX_POINT, 236, -1, -1, -1, -1, 3.750000e+02, 1.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 193, CONSTRAINT_TYPE_ID_FIX_POINT, 239, -1, -1, -1, -1, 4.750000e+02, 1.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 194, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 238, 234, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 195, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 233, 154, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 196, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 237, 230, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 197, CONSTRAINT_TYPE_ID_FIX_POINT, 240, -1, -1, -1, -1, 0.000000e+00, 7.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 198, CONSTRAINT_TYPE_ID_FIX_POINT, 243, -1, -1, -1, -1, 0.000000e+00, 1.750000e+02);
    err = jni_registerConstraintType(&env, eclass, 199, CONSTRAINT_TYPE_ID_FIX_POINT, 244, -1, -1, -1, -1, -7.500000e+01, 1.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 200, CONSTRAINT_TYPE_ID_FIX_POINT, 247, -1, -1, -1, -1, 2.500000e+01, 1.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 201, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 246, 242, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 202, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 241, 162, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 203, CONSTRAINT_TYPE_ID_FIX_POINT, 248, -1, -1, -1, -1, 5.000000e+01, 7.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 204, CONSTRAINT_TYPE_ID_FIX_POINT, 251, -1, -1, -1, -1, 5.000000e+01, 1.750000e+02);
    err = jni_registerConstraintType(&env, eclass, 205, CONSTRAINT_TYPE_ID_FIX_POINT, 252, -1, -1, -1, -1, -2.500000e+01, 1.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 206, CONSTRAINT_TYPE_ID_FIX_POINT, 255, -1, -1, -1, -1, 7.500000e+01, 1.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 207, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 254, 250, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 208, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 249, 170, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 209, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 253, 246, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 210, CONSTRAINT_TYPE_ID_FIX_POINT, 256, -1, -1, -1, -1, 1.000000e+02, 7.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 211, CONSTRAINT_TYPE_ID_FIX_POINT, 259, -1, -1, -1, -1, 1.000000e+02, 1.750000e+02);
    err = jni_registerConstraintType(&env, eclass, 212, CONSTRAINT_TYPE_ID_FIX_POINT, 260, -1, -1, -1, -1, 2.500000e+01, 1.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 213, CONSTRAINT_TYPE_ID_FIX_POINT, 263, -1, -1, -1, -1, 1.250000e+02, 1.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 214, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 262, 258, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 215, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 257, 178, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 216, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 261, 254, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 217, CONSTRAINT_TYPE_ID_FIX_POINT, 264, -1, -1, -1, -1, 1.500000e+02, 7.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 218, CONSTRAINT_TYPE_ID_FIX_POINT, 267, -1, -1, -1, -1, 1.500000e+02, 1.750000e+02);
    err = jni_registerConstraintType(&env, eclass, 219, CONSTRAINT_TYPE_ID_FIX_POINT, 268, -1, -1, -1, -1, 7.500000e+01, 1.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 220, CONSTRAINT_TYPE_ID_FIX_POINT, 271, -1, -1, -1, -1, 1.750000e+02, 1.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 221, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 270, 266, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 222, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 265, 186, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 223, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 269, 262, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 224, CONSTRAINT_TYPE_ID_FIX_POINT, 272, -1, -1, -1, -1, 2.000000e+02, 7.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 225, CONSTRAINT_TYPE_ID_FIX_POINT, 275, -1, -1, -1, -1, 2.000000e+02, 1.750000e+02);
    err = jni_registerConstraintType(&env, eclass, 226, CONSTRAINT_TYPE_ID_FIX_POINT, 276, -1, -1, -1, -1, 1.250000e+02, 1.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 227, CONSTRAINT_TYPE_ID_FIX_POINT, 279, -1, -1, -1, -1, 2.250000e+02, 1.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 228, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 278, 274, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 229, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 273, 194, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 230, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 277, 270, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 231, CONSTRAINT_TYPE_ID_FIX_POINT, 280, -1, -1, -1, -1, 2.500000e+02, 7.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 232, CONSTRAINT_TYPE_ID_FIX_POINT, 283, -1, -1, -1, -1, 2.500000e+02, 1.750000e+02);
    err = jni_registerConstraintType(&env, eclass, 233, CONSTRAINT_TYPE_ID_FIX_POINT, 284, -1, -1, -1, -1, 1.750000e+02, 1.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 234, CONSTRAINT_TYPE_ID_FIX_POINT, 287, -1, -1, -1, -1, 2.750000e+02, 1.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 235, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 286, 282, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 236, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 281, 202, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 237, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 285, 278, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 238, CONSTRAINT_TYPE_ID_FIX_POINT, 288, -1, -1, -1, -1, 3.000000e+02, 7.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 239, CONSTRAINT_TYPE_ID_FIX_POINT, 291, -1, -1, -1, -1, 3.000000e+02, 1.750000e+02);
    err = jni_registerConstraintType(&env, eclass, 240, CONSTRAINT_TYPE_ID_FIX_POINT, 292, -1, -1, -1, -1, 2.250000e+02, 1.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 241, CONSTRAINT_TYPE_ID_FIX_POINT, 295, -1, -1, -1, -1, 3.250000e+02, 1.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 242, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 294, 290, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 243, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 289, 210, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 244, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 293, 286, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 245, CONSTRAINT_TYPE_ID_FIX_POINT, 296, -1, -1, -1, -1, 3.500000e+02, 7.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 246, CONSTRAINT_TYPE_ID_FIX_POINT, 299, -1, -1, -1, -1, 3.500000e+02, 1.750000e+02);
    err = jni_registerConstraintType(&env, eclass, 247, CONSTRAINT_TYPE_ID_FIX_POINT, 300, -1, -1, -1, -1, 2.750000e+02, 1.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 248, CONSTRAINT_TYPE_ID_FIX_POINT, 303, -1, -1, -1, -1, 3.750000e+02, 1.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 249, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 302, 298, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 250, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 297, 218, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 251, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 301, 294, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 252, CONSTRAINT_TYPE_ID_FIX_POINT, 304, -1, -1, -1, -1, 4.000000e+02, 7.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 253, CONSTRAINT_TYPE_ID_FIX_POINT, 307, -1, -1, -1, -1, 4.000000e+02, 1.750000e+02);
    err = jni_registerConstraintType(&env, eclass, 254, CONSTRAINT_TYPE_ID_FIX_POINT, 308, -1, -1, -1, -1, 3.250000e+02, 1.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 255, CONSTRAINT_TYPE_ID_FIX_POINT, 311, -1, -1, -1, -1, 4.250000e+02, 1.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 256, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 310, 306, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 257, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 305, 226, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 258, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 309, 302, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 259, CONSTRAINT_TYPE_ID_FIX_POINT, 312, -1, -1, -1, -1, 4.500000e+02, 7.500000e+01);
    err = jni_registerConstraintType(&env, eclass, 260, CONSTRAINT_TYPE_ID_FIX_POINT, 315, -1, -1, -1, -1, 4.500000e+02, 1.750000e+02);
    err = jni_registerConstraintType(&env, eclass, 261, CONSTRAINT_TYPE_ID_FIX_POINT, 316, -1, -1, -1, -1, 3.750000e+02, 1.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 262, CONSTRAINT_TYPE_ID_FIX_POINT, 319, -1, -1, -1, -1, 4.750000e+02, 1.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 263, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 318, 314, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 264, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 313, 234, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 265, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 317, 310, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 266, CONSTRAINT_TYPE_ID_FIX_POINT, 320, -1, -1, -1, -1, 0.000000e+00, 1.250000e+02);
    err = jni_registerConstraintType(&env, eclass, 267, CONSTRAINT_TYPE_ID_FIX_POINT, 323, -1, -1, -1, -1, 0.000000e+00, 2.250000e+02);
    err = jni_registerConstraintType(&env, eclass, 268, CONSTRAINT_TYPE_ID_FIX_POINT, 324, -1, -1, -1, -1, -7.500000e+01, 2.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 269, CONSTRAINT_TYPE_ID_FIX_POINT, 327, -1, -1, -1, -1, 2.500000e+01, 2.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 270, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 326, 322, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 271, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 321, 242, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 272, CONSTRAINT_TYPE_ID_FIX_POINT, 328, -1, -1, -1, -1, 5.000000e+01, 1.250000e+02);
    err = jni_registerConstraintType(&env, eclass, 273, CONSTRAINT_TYPE_ID_FIX_POINT, 331, -1, -1, -1, -1, 5.000000e+01, 2.250000e+02);
    err = jni_registerConstraintType(&env, eclass, 274, CONSTRAINT_TYPE_ID_FIX_POINT, 332, -1, -1, -1, -1, -2.500000e+01, 2.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 275, CONSTRAINT_TYPE_ID_FIX_POINT, 335, -1, -1, -1, -1, 7.500000e+01, 2.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 276, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 334, 330, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 277, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 329, 250, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 278, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 333, 326, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 279, CONSTRAINT_TYPE_ID_FIX_POINT, 336, -1, -1, -1, -1, 1.000000e+02, 1.250000e+02);
    err = jni_registerConstraintType(&env, eclass, 280, CONSTRAINT_TYPE_ID_FIX_POINT, 339, -1, -1, -1, -1, 1.000000e+02, 2.250000e+02);
    err = jni_registerConstraintType(&env, eclass, 281, CONSTRAINT_TYPE_ID_FIX_POINT, 340, -1, -1, -1, -1, 2.500000e+01, 2.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 282, CONSTRAINT_TYPE_ID_FIX_POINT, 343, -1, -1, -1, -1, 1.250000e+02, 2.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 283, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 342, 338, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 284, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 337, 258, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 285, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 341, 334, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 286, CONSTRAINT_TYPE_ID_FIX_POINT, 344, -1, -1, -1, -1, 1.500000e+02, 1.250000e+02);
    err = jni_registerConstraintType(&env, eclass, 287, CONSTRAINT_TYPE_ID_FIX_POINT, 347, -1, -1, -1, -1, 1.500000e+02, 2.250000e+02);
    err = jni_registerConstraintType(&env, eclass, 288, CONSTRAINT_TYPE_ID_FIX_POINT, 348, -1, -1, -1, -1, 7.500000e+01, 2.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 289, CONSTRAINT_TYPE_ID_FIX_POINT, 351, -1, -1, -1, -1, 1.750000e+02, 2.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 290, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 350, 346, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 291, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 345, 266, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 292, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 349, 342, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 293, CONSTRAINT_TYPE_ID_FIX_POINT, 352, -1, -1, -1, -1, 2.000000e+02, 1.250000e+02);
    err = jni_registerConstraintType(&env, eclass, 294, CONSTRAINT_TYPE_ID_FIX_POINT, 355, -1, -1, -1, -1, 2.000000e+02, 2.250000e+02);
    err = jni_registerConstraintType(&env, eclass, 295, CONSTRAINT_TYPE_ID_FIX_POINT, 356, -1, -1, -1, -1, 1.250000e+02, 2.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 296, CONSTRAINT_TYPE_ID_FIX_POINT, 359, -1, -1, -1, -1, 2.250000e+02, 2.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 297, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 358, 354, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 298, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 353, 274, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 299, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 357, 350, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 300, CONSTRAINT_TYPE_ID_FIX_POINT, 360, -1, -1, -1, -1, 2.500000e+02, 1.250000e+02);
    err = jni_registerConstraintType(&env, eclass, 301, CONSTRAINT_TYPE_ID_FIX_POINT, 363, -1, -1, -1, -1, 2.500000e+02, 2.250000e+02);
    err = jni_registerConstraintType(&env, eclass, 302, CONSTRAINT_TYPE_ID_FIX_POINT, 364, -1, -1, -1, -1, 1.750000e+02, 2.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 303, CONSTRAINT_TYPE_ID_FIX_POINT, 367, -1, -1, -1, -1, 2.750000e+02, 2.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 304, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 366, 362, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 305, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 361, 282, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 306, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 365, 358, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 307, CONSTRAINT_TYPE_ID_FIX_POINT, 368, -1, -1, -1, -1, 3.000000e+02, 1.250000e+02);
    err = jni_registerConstraintType(&env, eclass, 308, CONSTRAINT_TYPE_ID_FIX_POINT, 371, -1, -1, -1, -1, 3.000000e+02, 2.250000e+02);
    err = jni_registerConstraintType(&env, eclass, 309, CONSTRAINT_TYPE_ID_FIX_POINT, 372, -1, -1, -1, -1, 2.250000e+02, 2.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 310, CONSTRAINT_TYPE_ID_FIX_POINT, 375, -1, -1, -1, -1, 3.250000e+02, 2.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 311, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 374, 370, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 312, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 369, 290, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 313, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 373, 366, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 314, CONSTRAINT_TYPE_ID_FIX_POINT, 376, -1, -1, -1, -1, 3.500000e+02, 1.250000e+02);
    err = jni_registerConstraintType(&env, eclass, 315, CONSTRAINT_TYPE_ID_FIX_POINT, 379, -1, -1, -1, -1, 3.500000e+02, 2.250000e+02);
    err = jni_registerConstraintType(&env, eclass, 316, CONSTRAINT_TYPE_ID_FIX_POINT, 380, -1, -1, -1, -1, 2.750000e+02, 2.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 317, CONSTRAINT_TYPE_ID_FIX_POINT, 383, -1, -1, -1, -1, 3.750000e+02, 2.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 318, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 382, 378, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 319, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 377, 298, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 320, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 381, 374, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 321, CONSTRAINT_TYPE_ID_FIX_POINT, 384, -1, -1, -1, -1, 4.000000e+02, 1.250000e+02);
    err = jni_registerConstraintType(&env, eclass, 322, CONSTRAINT_TYPE_ID_FIX_POINT, 387, -1, -1, -1, -1, 4.000000e+02, 2.250000e+02);
    err = jni_registerConstraintType(&env, eclass, 323, CONSTRAINT_TYPE_ID_FIX_POINT, 388, -1, -1, -1, -1, 3.250000e+02, 2.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 324, CONSTRAINT_TYPE_ID_FIX_POINT, 391, -1, -1, -1, -1, 4.250000e+02, 2.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 325, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 390, 386, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 326, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 385, 306, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 327, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 389, 382, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 328, CONSTRAINT_TYPE_ID_FIX_POINT, 392, -1, -1, -1, -1, 4.500000e+02, 1.250000e+02);
    err = jni_registerConstraintType(&env, eclass, 329, CONSTRAINT_TYPE_ID_FIX_POINT, 395, -1, -1, -1, -1, 4.500000e+02, 2.250000e+02);
    err = jni_registerConstraintType(&env, eclass, 330, CONSTRAINT_TYPE_ID_FIX_POINT, 396, -1, -1, -1, -1, 3.750000e+02, 2.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 331, CONSTRAINT_TYPE_ID_FIX_POINT, 399, -1, -1, -1, -1, 4.750000e+02, 2.000000e+02);
    err = jni_registerConstraintType(&env, eclass, 332, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 398, 394, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 333, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 393, 314, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 334, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 397, 390, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 335, CONSTRAINT_TYPE_ID_FIX_POINT, 400, -1, -1, -1, -1, 0.000000e+00, 1.750000e+02);
    err = jni_registerConstraintType(&env, eclass, 336, CONSTRAINT_TYPE_ID_FIX_POINT, 403, -1, -1, -1, -1, 0.000000e+00, 2.750000e+02);
    err = jni_registerConstraintType(&env, eclass, 337, CONSTRAINT_TYPE_ID_FIX_POINT, 404, -1, -1, -1, -1, -7.500000e+01, 2.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 338, CONSTRAINT_TYPE_ID_FIX_POINT, 407, -1, -1, -1, -1, 2.500000e+01, 2.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 339, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 406, 402, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 340, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 401, 322, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 341, CONSTRAINT_TYPE_ID_FIX_POINT, 408, -1, -1, -1, -1, 5.000000e+01, 1.750000e+02);
    err = jni_registerConstraintType(&env, eclass, 342, CONSTRAINT_TYPE_ID_FIX_POINT, 411, -1, -1, -1, -1, 5.000000e+01, 2.750000e+02);
    err = jni_registerConstraintType(&env, eclass, 343, CONSTRAINT_TYPE_ID_FIX_POINT, 412, -1, -1, -1, -1, -2.500000e+01, 2.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 344, CONSTRAINT_TYPE_ID_FIX_POINT, 415, -1, -1, -1, -1, 7.500000e+01, 2.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 345, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 414, 410, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 346, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 409, 330, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 347, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 413, 406, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 348, CONSTRAINT_TYPE_ID_FIX_POINT, 416, -1, -1, -1, -1, 1.000000e+02, 1.750000e+02);
    err = jni_registerConstraintType(&env, eclass, 349, CONSTRAINT_TYPE_ID_FIX_POINT, 419, -1, -1, -1, -1, 1.000000e+02, 2.750000e+02);
    err = jni_registerConstraintType(&env, eclass, 350, CONSTRAINT_TYPE_ID_FIX_POINT, 420, -1, -1, -1, -1, 2.500000e+01, 2.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 351, CONSTRAINT_TYPE_ID_FIX_POINT, 423, -1, -1, -1, -1, 1.250000e+02, 2.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 352, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 422, 418, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 353, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 417, 338, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 354, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 421, 414, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 355, CONSTRAINT_TYPE_ID_FIX_POINT, 424, -1, -1, -1, -1, 1.500000e+02, 1.750000e+02);
    err = jni_registerConstraintType(&env, eclass, 356, CONSTRAINT_TYPE_ID_FIX_POINT, 427, -1, -1, -1, -1, 1.500000e+02, 2.750000e+02);
    err = jni_registerConstraintType(&env, eclass, 357, CONSTRAINT_TYPE_ID_FIX_POINT, 428, -1, -1, -1, -1, 7.500000e+01, 2.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 358, CONSTRAINT_TYPE_ID_FIX_POINT, 431, -1, -1, -1, -1, 1.750000e+02, 2.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 359, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 430, 426, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 360, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 425, 346, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 361, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 429, 422, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 362, CONSTRAINT_TYPE_ID_FIX_POINT, 432, -1, -1, -1, -1, 2.000000e+02, 1.750000e+02);
    err = jni_registerConstraintType(&env, eclass, 363, CONSTRAINT_TYPE_ID_FIX_POINT, 435, -1, -1, -1, -1, 2.000000e+02, 2.750000e+02);
    err = jni_registerConstraintType(&env, eclass, 364, CONSTRAINT_TYPE_ID_FIX_POINT, 436, -1, -1, -1, -1, 1.250000e+02, 2.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 365, CONSTRAINT_TYPE_ID_FIX_POINT, 439, -1, -1, -1, -1, 2.250000e+02, 2.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 366, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 438, 434, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 367, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 433, 354, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 368, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 437, 430, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 369, CONSTRAINT_TYPE_ID_FIX_POINT, 440, -1, -1, -1, -1, 2.500000e+02, 1.750000e+02);
    err = jni_registerConstraintType(&env, eclass, 370, CONSTRAINT_TYPE_ID_FIX_POINT, 443, -1, -1, -1, -1, 2.500000e+02, 2.750000e+02);
    err = jni_registerConstraintType(&env, eclass, 371, CONSTRAINT_TYPE_ID_FIX_POINT, 444, -1, -1, -1, -1, 1.750000e+02, 2.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 372, CONSTRAINT_TYPE_ID_FIX_POINT, 447, -1, -1, -1, -1, 2.750000e+02, 2.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 373, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 446, 442, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 374, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 441, 362, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 375, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 445, 438, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 376, CONSTRAINT_TYPE_ID_FIX_POINT, 448, -1, -1, -1, -1, 3.000000e+02, 1.750000e+02);
    err = jni_registerConstraintType(&env, eclass, 377, CONSTRAINT_TYPE_ID_FIX_POINT, 451, -1, -1, -1, -1, 3.000000e+02, 2.750000e+02);
    err = jni_registerConstraintType(&env, eclass, 378, CONSTRAINT_TYPE_ID_FIX_POINT, 452, -1, -1, -1, -1, 2.250000e+02, 2.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 379, CONSTRAINT_TYPE_ID_FIX_POINT, 455, -1, -1, -1, -1, 3.250000e+02, 2.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 380, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 454, 450, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 381, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 449, 370, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 382, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 453, 446, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 383, CONSTRAINT_TYPE_ID_FIX_POINT, 456, -1, -1, -1, -1, 3.500000e+02, 1.750000e+02);
    err = jni_registerConstraintType(&env, eclass, 384, CONSTRAINT_TYPE_ID_FIX_POINT, 459, -1, -1, -1, -1, 3.500000e+02, 2.750000e+02);
    err = jni_registerConstraintType(&env, eclass, 385, CONSTRAINT_TYPE_ID_FIX_POINT, 460, -1, -1, -1, -1, 2.750000e+02, 2.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 386, CONSTRAINT_TYPE_ID_FIX_POINT, 463, -1, -1, -1, -1, 3.750000e+02, 2.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 387, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 462, 458, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 388, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 457, 378, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 389, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 461, 454, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 390, CONSTRAINT_TYPE_ID_FIX_POINT, 464, -1, -1, -1, -1, 4.000000e+02, 1.750000e+02);
    err = jni_registerConstraintType(&env, eclass, 391, CONSTRAINT_TYPE_ID_FIX_POINT, 467, -1, -1, -1, -1, 4.000000e+02, 2.750000e+02);
    err = jni_registerConstraintType(&env, eclass, 392, CONSTRAINT_TYPE_ID_FIX_POINT, 468, -1, -1, -1, -1, 3.250000e+02, 2.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 393, CONSTRAINT_TYPE_ID_FIX_POINT, 471, -1, -1, -1, -1, 4.250000e+02, 2.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 394, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 470, 466, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 395, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 465, 386, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 396, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 469, 462, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 397, CONSTRAINT_TYPE_ID_FIX_POINT, 472, -1, -1, -1, -1, 4.500000e+02, 1.750000e+02);
    err = jni_registerConstraintType(&env, eclass, 398, CONSTRAINT_TYPE_ID_FIX_POINT, 475, -1, -1, -1, -1, 4.500000e+02, 2.750000e+02);
    err = jni_registerConstraintType(&env, eclass, 399, CONSTRAINT_TYPE_ID_FIX_POINT, 476, -1, -1, -1, -1, 3.750000e+02, 2.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 400, CONSTRAINT_TYPE_ID_FIX_POINT, 479, -1, -1, -1, -1, 4.750000e+02, 2.500000e+02);
    err = jni_registerConstraintType(&env, eclass, 401, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 478, 474, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 402, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 473, 394, -1, -1, -1, 0.000000e+00, 0.000000e+00);
    err = jni_registerConstraintType(&env, eclass, 403, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 477, 470, -1, -1, -1, 0.000000e+00, 0.000000e+00);

    /// --definition-end: 



    return err;
}

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
    
    settings::COMPUTATION_MODE.reset((int)ComputationMode::SPARSE_MODE);                // settings::COMPUTATION_MODE = [ 1 - DENSE_MODE , 2 - SPARSE_MODE , *3 - DIRECT_MODE , 4 - COMPACT_MODE ]
    settings::SOLVER_MODE.reset((int)SolverMode::SPARSE_QR);                            // settings::SOLVER_MODE  = [ DENSE_LU = 1, SPARSE_QR = 2, SPARSE_ILU = 3, ] 
        
    //jni_setLongProperty(&env, eclass, 20, (jlong) 2);  // settings::COMPUTATION_MODE = [ 1 - DENSE_MODE , 2 - SPARSE_MODE , *3 - DIRECT_MODE , 4 - COMPACT_MODE ]

    //jni_setLongProperty(&env, eclass, 21, (jlong) 3); // settings::SOLVER_MODE  = [ DENSE_LU = 1, SPARSE_QR = 2, SPARSE_ILU = 3, ]


    settings::DEBUG_SOLVER_CONVERGENCE.reset(true); // jni_setBooleanProperty(&env, eclass, 8, (jboolean) true);  // settings::DEBUG_SOLVER_CONVERGENCE = false;
    settings::DEBUG_CHECK_ARG.reset(true);          // jni_setBooleanProperty(&env, eclass, 9, (jboolean) true);  // settings::DEBUG_CHECK_ARG = false;
    settings::DEBUG_TENSOR_A.reset(false);           // jni_setBooleanProperty(&env, eclass, 1, (jboolean) false);  // settings::DEBUG_TENSOR_A= true;
    settings::DEBUG_TENSOR_B.reset(false);           // jni_setBooleanProperty(&env, eclass, 2, (jboolean) false);  // settings::DEBUG_TENSOR_B= true;

    settings::DEBUG.reset(true);                   // jni_setBooleanProperty(&env, eclass, 3, (jboolean) false); // settings::DEBUG= true;
    settings::DEBUG_TENSOR_SV.reset(false);         // jni_setBooleanProperty(&env, eclass, 4, (jboolean) false); // settings::DEBUG_TENSOR_SV= true;

    settings::SOLVER_INC_HESSIAN.reset(false);      // jni_setBooleanProperty(&env, eclass, 24, (jboolean) true);  // ;
    settings::DEBUG_CSR_FORMAT.reset(false);        // jni_setBooleanProperty(&env, eclass, 30, (jboolean) false); // ;
    settings::DEBUG_COO_FORMAT.reset(false);        // jni_setBooleanProperty(&env, eclass, 31, (jboolean) false); // ;

    err = jni_initComputationContext(&env, eclass);
    /// ----------------------------------------------------------------------------------------------------- //
    
    //model_single_line(env, eclass);

    model_circle_line_tangetn_perpendicular(env, eclass);

    //model_6x10(env, eclass);
      

    /// ----------------------------------------------------------------------------------------------------- //

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
