#include <iostream>
#include <stdio.h>

#include "com_mstruzek_jni_JNISolverGate.h"
#define jni_initDriver                 Java_com_mstruzek_jni_JNISolverGate_initDriver
#define jni_getLastError               Java_com_mstruzek_jni_JNISolverGate_getLastError
#define jni_closeDriver                Java_com_mstruzek_jni_JNISolverGate_closeDriver
#define jni_resetDatabase              Java_com_mstruzek_jni_JNISolverGate_resetDatabase
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


int main(int argc, char* args[]) 
{
    printf("empty inspector \n");

    return 0;
}