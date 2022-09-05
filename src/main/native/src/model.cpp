#include "model.cuh"

#include <stdio.h>
#include <stdlib.h>

namespace graph {

int constraintSize(Constraint const &constraint) {
    switch (constraint.constraintTypeId) {
    case CONSTRAINT_TYPE_ID_FIX_POINT:
        return 2;
    case CONSTRAINT_TYPE_ID_PARAMETRIZED_XFIX:
        return 1;
    case CONSTRAINT_TYPE_ID_PARAMETRIZED_YFIX:
        return 1;
    case CONSTRAINT_TYPE_ID_CONNECT_2_POINTS:
        return 2;
    case CONSTRAINT_TYPE_ID_HORIZONTAL_POINT:
        return 1;
    case CONSTRAINT_TYPE_ID_VERTICAL_POINT:
        return 1;
    case CONSTRAINT_TYPE_ID_LINES_PARALLELISM:
        return 1;
    case CONSTRAINT_TYPE_ID_LINES_PERPENDICULAR:
        return 1;
    case CONSTRAINT_TYPE_ID_EQUAL_LENGTH:
        return 1;
    case CONSTRAINT_TYPE_ID_PARAMETRIZED_LENGTH:
        return 1;
    case CONSTRAINT_TYPE_ID_TANGENCY:
        return 1;
    case CONSTRAINT_TYPE_ID_CIRCLE_TANGENCY:
        return 1;
    case CONSTRAINT_TYPE_ID_DISTANCE_2_POINTS:
        return 1;
    case CONSTRAINT_TYPE_ID_DISTANCE_POINT_LINE:
        return 1;
    case CONSTRAINT_TYPE_ID_ANGLE_2_LINES:
        return 1;
    case CONSTRAINT_TYPE_ID_SET_HORIZONTAL:
        return 1;
    case CONSTRAINT_TYPE_ID_SET_VERTICAL:
        return 1;

    default:
        printf("unknown constraint type");
        exit(1);
    }
}

int geometricSetSize(Geometric const &geometric) {
    switch (geometric.geometricTypeId) {
    case GEOMETRIC_TYPE_ID_FREE_POINT:
        return 3 * 2;
    case GEOMETRIC_TYPE_ID_LINE:
        return 4 * 2;
    case GEOMETRIC_TYPE_ID_CIRCLE:
        return 4 * 2;
    case GEOMETRIC_TYPE_ID_ARC:
        return 7 * 2;
    default:
        printf("unknown geometric type");
        exit(1);
    }
}

///
/// __device__ __host__ COO tensor format requirments
///
int tensorOpsCooStiffnesCoefficients(Geometric const &geometric) {
    switch (geometric.geometricTypeId) {
    case GEOMETRIC_TYPE_ID_FREE_POINT:
        return 7 * 2; // 7 - plusSubTensor  * diagonal I (2)
    case GEOMETRIC_TYPE_ID_LINE:
        return 10 * 2; // 10 - plusSubTensor * diagonal I (2)
    case GEOMETRIC_TYPE_ID_CIRCLE:
        return 10 * 2; // 10 - plusSubTensor * diagonal I (2)
    case GEOMETRIC_TYPE_ID_ARC:
        return 19 * 2; // 19 - plusSubTensor * diagonal I (2)
    default:
        printf("unknown geometric type");
        exit(1);
    }
}

///
/// __device__ __host__ ??  ------ COO tensor format requirments --> exclusive_scan --> cudaMalloc(accWriteOffset)
///
int tensorOpsCooConstraintJacobian(Constraint const &constraint) {
    switch (constraint.constraintTypeId) {
    case CONSTRAINT_TYPE_ID_FIX_POINT:
        /// 1 * diagonal (2)
        return 2;
    case CONSTRAINT_TYPE_ID_PARAMETRIZED_XFIX:
        ///  1 * quick
        return 1;
    case CONSTRAINT_TYPE_ID_PARAMETRIZED_YFIX:
        ///  1 * quick
        return 1;
    case CONSTRAINT_TYPE_ID_CONNECT_2_POINTS:
        /// 2 * diagonal (2)
        return 4;
    case CONSTRAINT_TYPE_ID_HORIZONTAL_POINT:
        /// 2 * quick
        return 2;
    case CONSTRAINT_TYPE_ID_VERTICAL_POINT:
        ///  2 * quick
        return 2;
    case CONSTRAINT_TYPE_ID_LINES_PARALLELISM:
        /// 4 * vector(2)
        return 8;
    case CONSTRAINT_TYPE_ID_LINES_PERPENDICULAR:
        /// 4 * vector(2)
        return 8;
    case CONSTRAINT_TYPE_ID_EQUAL_LENGTH:
        /// 4 * vector(2)
        return 8;
    case CONSTRAINT_TYPE_ID_PARAMETRIZED_LENGTH:
        /// 4 * vector(2)
        return 8;
    case CONSTRAINT_TYPE_ID_TANGENCY:
        /// 4 * vector(2)
        return 8;
    case CONSTRAINT_TYPE_ID_CIRCLE_TANGENCY:
        /// 4 * vector(2)
        return 8;
    case CONSTRAINT_TYPE_ID_DISTANCE_2_POINTS:
        /// 2 * vector(2)
        return 4;        
    case CONSTRAINT_TYPE_ID_DISTANCE_POINT_LINE:
        /// 3 * vector(2)
        return 6;
    case CONSTRAINT_TYPE_ID_ANGLE_2_LINES:
        /// 4 * vector(2)
        return 8;
    case CONSTRAINT_TYPE_ID_SET_HORIZONTAL:
        /// 2 * vector(2)
        return 4;
    case CONSTRAINT_TYPE_ID_SET_VERTICAL:
        /// 2 * vector(2)
        return 4;
    default:
        printf("unknown constraint type");
        exit(1);
    }
}

} // namespace graph