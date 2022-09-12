#ifndef __COMPUTATION_STATE_H_ 
#define __COMPUTATION_STATE_H_

#include "device_launch_parameters.h"

#include "model.cuh"

///  ===============================================================================

#ifndef M_PI
#define M_PI 3.14159265358979323846 // pi
#endif

struct ComputationState {  

    ComputationState() : INV_P(NULL), cooRowInd(NULL), cooColInd(NULL), cooVal(NULL), singularity(-1) 
    {}

#ifdef __NVCC__
    

    __GPU_DEV_INL__ graph::Vector const &getPoint(int pointId) const {
        int offset = pointOffset[pointId];
        graph::Vector *vector;
        *((void **)&vector) = &SV[offset * 2];
        return *vector;
    }

    __GPU_DEV_INL__ double getLagrangeMultiplier(int constraintId) const {
        int multiOffset = accConstraintSize[constraintId];
        return SV[size + multiOffset];
    }

    __GPU_DEV_INL__ graph::Point const &getPointRef(int pointId) {
        const size_t offset = pointOffset[pointId];
        return points[offset];
    }

    __GPU_DEV_INL__ graph::Geometric *getGeometricObject(int geometricId) {
        const size_t offset = geometricOffset[geometricId];
        return &geometrics[offset];
    }

    __GPU_DEV_INL__ graph::Constraint *getConstraint(int constraintId) {
        const size_t offset = constraintOffset[constraintId];
        return &constraints[offset];
    }

    __GPU_DEV_INL__ graph::Parameter *getParameter(int parameterId) {
        const size_t offset = parameterOffset[parameterId];
        return &parameters[offset];
    }

#endif

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

    NVector<graph::Point> points;
    NVector<graph::Geometric> geometrics;
    NVector<graph::Constraint> constraints;
    NVector<graph::Parameter> parameters;

    // NVector direct map
    int *pointOffset;

    // NVector direct map
    int *geometricOffset;

    // NVector direct map
    int *constraintOffset;

    /// paramater offs from given ID
    int *parameterOffset;

    /// Accumulative offs with geometric size evaluation function,  [ 0, ... , N ]
    int *accGeometricSize;

    /// Accumulative offs with constraint size evaluation function, [ 0, ... , N ]
    int *accConstraintSize;

    /// computation mode applied onto "A tensor" at this iteration
    ComputationMode computationMode;

    /// Relative Offsets - Accumulated Writes in COO format from kernel into  Stiff Tensor
    int *accCooWriteStiffTensor;

    /// Relative Offsets - Accumulated Writes in COO format from kernel into Jacobian Tensor
    int *accCooWriteJacobianTensor;
    
    /// Relative Offsets -  Accumulate writes in COO format form kernel into Hessian Tensor 
    int *accCooWriteHessianTensor;

    /// non-zero elements in coo or csr ; nnz =  cooWritesStiffSize + 2 * cooWirtesJacobianSize + optional(hessian)*
    int nnz;

    /// offset for Jacobian kernel Writes
    int cooWritesStiffSize;

    /// offset for Transposed Jacobian kernel writes
    int cooWirtesJacobianSize;

    // CSR format, inverse permutation vector, direct addressing from COO to sparse matrix in CSR format
    int *INV_P;

    /// not-transformed row vector of indicies, Coordinate Format COO
    int *cooRowInd;

    /// not-transformed column vector of indicies, Coordinate Format COO
    int *cooColInd;

    /// COO vector of values, Coordinate Format COO, or CSR format sorted
    double *cooVal;

    /// tensor A solver response - -1 invertible otherwise exect index of diagonal element 
    int singularity;

};



#endif // __COMPUTATION_STATE_H_
