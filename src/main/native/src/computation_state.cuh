#include "computation_state_data.h"

///  ===============================================================================

#ifndef M_PI
#define M_PI 3.14159265358979323846 // pi
#endif

struct ComputationState : public ComputationStateData {

    __host__ __device__ graph::Vector const &getPoint(int pointId) const {
        int offset = pointOffset[pointId];
        graph::Vector *vector;
        *((void **)&vector) = &SV[offset * 2];
        return *vector;
    }

    __host__ __device__ double getLagrangeMultiplier(int constraintId) const {
        int multiOffset = accConstraintSize[constraintId];
        return SV[size + multiOffset];
    }

    __host__ __device__  graph::Point const &getPointRef(int pointId) const {
        int offset = pointOffset[pointId];
        return points[offset];
    }

    __host__ __device__  graph::Geometric *getGeometricObject(int geometricId) const {
        int offset = geometricOffset[geometricId];
        return &geometrics[offset];
    }

    __host__ __device__ graph::Constraint *getConstraint(int constraintId) const {
        int offset = constraintOffset[constraintId];
        return &constraints[offset];
    }

    __host__ __device__ graph::Parameter *getParameter(int parameterId) const {
        int offset = parameterOffset[parameterId];
        return &parameters[offset];
    }

};

__host__ __device__ __forceinline__ double toRadians(double value) { return (M_PI / 180.0) * value; }
