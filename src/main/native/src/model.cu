#include "model.h"

#define DEGREES_TO_RADIANS 0.017453292519943295;


/// 
/// 
///  Configuration Properties 
///  -> CUDA C/C++ 
///  -> Common 
///  -> Generate Relocatable Device Code -> Yes (-rdc=true)
/// 
/// 

namespace graph { 


__host__ 
__device__ double toRadians(double angdeg) { return angdeg * DEGREES_TO_RADIANS; }


/// Tensor


__host__ 
__device__ Tensor::Tensor() : nonMemOwning(true){};


__host__ 
__device__ Tensor::Tensor(bool nonMemOwning) : nonMemOwning(nonMemOwning){};


__host__ 
__device__ void Tensor::setVector(int offsetRow, int offsetCol, Vector const &value){ 
    this->cols = offsetRow;
    /// ALL - dwa przypadki na miejscu
};

__host__ 
__device__ void Tensor::plusSubTensor(int offsetRow, int offsetCol, Tensor const &mt) { 
    
    return;
}

__host__ 
__device__ void Tensor::setSubTensor(int offsetRow, int offsetCol, Tensor const &mt) {

        /// ALL - dwa przypadki na miejscu
        return;
}

__host__ 
__device__ Tensor Tensor::multiplyC(double scalar) {
        /// SmallTensor
        return Tensor(false);
}

__host__ 
__device__ Tensor Tensor::plus(Tensor const &other) {


        return Tensor(false);
}

__host__ 
__device__ Tensor Tensor::transpose() const { 
    return Tensor(false); 
}

/// Small-Tensor

__host__ 
__device__ SmallTensor::SmallTensor(double a00, double a11) : Tensor(true) {
        tensor[0] = a00;
        tensor[3] = a11;
}

__host__ 
__device__ SmallTensor::SmallTensor(double a00, double a01, double a10, double a11) : SmallTensor() {
        tensor[0] = a00;
        tensor[1] = a01;
        tensor[2] = a10;
        tensor[3] = a11;
}

__host__ 
__device__ SmallTensor SmallTensor::diagonal(double diagonal) { return SmallTensor(diagonal, diagonal); }

__host__ 
__device__ SmallTensor SmallTensor::tensorR() {
        double a00 = 0.0;
        double a01 = -1.0;
        double a10 = 1.0;
        double a11 = 0.0;
        return SmallTensor(a00, a01, a10, a11);
}

/// Vector

/// Point

/// Constraint

/// Parameter

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
                return -1;
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
                return -1;
        }
}




} // namespace graph