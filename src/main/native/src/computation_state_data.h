#pragma once

#include "model.cuh"

struct ComputationStateData {

    /// computation unique id
    int cID;     
    
    /// device variable cuBlas
    int info;    
    
    /// cublasDnrm2(...)
    double norm; 

    double *A;

    /// State Vector  [ SV = SV + dx ] , previous task -- "lineage"
    double *SV; 
    
    /// przyrosty   [ A * dx = b ]
    double *dx; 
    
    ///  right hand side vector - these are Fi / Fq
    double *b;

    ///  eventually synchronized into 'norm field on the host
    double *dev_norm; 

    /// wektor stanu geometric structure - effectievly consts
    size_t size;      

    /// wspolczynniki Lagrange
    size_t coffSize;  

    /// N - dimension = size + coffSize
    size_t dimension; 

    graph::Point *points;
    graph::Geometric *geometrics;
    graph::Constraint *constraints;
    graph::Parameter *parameters;

    const int *pointOffset;       

    const int *constraintOffset;  

    /// paramater offs from given ID
    const int *parameterOffset;   

    /// accumulative offs with geometric size evaluation function
    const int *accGeometricSize;  

    /// accumulative offs with constraint size evaluation function
    const int *accConstraintSize; 
};

