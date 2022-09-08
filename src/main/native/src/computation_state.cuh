#include "computation_state_data.h"

///  ===============================================================================

#ifndef M_PI
#define M_PI 3.14159265358979323846 // pi
#endif

struct ComputationState : public ComputationStateData {

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

};

__GPU_DEV_INL__ double toRadians(double value) { return (M_PI / 180.0) * value; }
