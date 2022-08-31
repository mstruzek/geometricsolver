#include "computation_state_data.h"

///  ===============================================================================

#ifndef M_PI
#define M_PI 3.14159265358979323846 // pi
#endif

struct ComputationState : public ComputationStateData {

    __host__ __device__ __forceinline__ graph::Vector const &getPoint(int pointId) const {
        int offset = pointOffset[pointId];
        graph::Vector *vector;
        *((void **)&vector) = &SV[offset * 2];
        return *vector;
    }

    __host__ __device__ __forceinline__ double getLagrangeMultiplier(int constraintId) const {
        int multiOffset = accConstraintSize[constraintId];
        return SV[size + multiOffset];
    }

    __host__ __device__ __forceinline__ graph::Point const &getPointRef(int pointId) const {
        int offset = pointOffset[pointId];
        return points[offset];
    }

    __host__ __device__ __forceinline__ graph::Geometric *getGeometricObject(int geometricId) const {
        /// geometricId is associated with `threadIdx
        return static_cast<graph::Geometric *>(&geometrics[geometricId]);
    }

    __host__ __device__ __forceinline__ graph::Constraint *getConstraint(int constraintId) const {
        /// constraintId is associated with `threadIdx
        return static_cast<graph::Constraint *>(&constraints[constraintId]);
    }

    __host__ __device__ __forceinline__ graph::Parameter *getParameter(int parameterId) const {
        int offset = parameterOffset[parameterId];
        return static_cast<graph::Parameter *>(&parameters[offset]);
    }

};

__host__ __device__ __forceinline__ double toRadians(double value) { return (M_PI / 180.0) * value; }
