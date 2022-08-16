#ifndef _MODEL_H_
#define _MODEL_H_

#include <cuda_runtime_api.h>
#include <math.h>

#include "model_config.h"

namespace graph {


class Vector;

struct Geometric;

struct Constraint;


int constraintSize(Constraint const &constraint);
int geometricSetSize(Geometric const &geometric);



#define DEGREES_TO_RADIANS 0.017453292519943295;

__host__ __device__ double toRadians(double angdeg) { 
    return angdeg * DEGREES_TO_RADIANS; 
}



/// Vector

/// Point

/// Constraint

/// Parameter

class Tensor {
      public:

        __host__ __device__ Tensor() : nonMemOwning(true)
        {};

        __host__ __device__ Tensor(bool nonMemOwning) : nonMemOwning(nonMemOwning)
        {};

        __host__ __device__ void setVector(int offsetRow, int offsetCol, Vector const &value) {
                this->cols = offsetRow;
        };

        __host__ __device__ void plusSubTensor(int offsetRow, int offsetCol, Tensor const &mt) { return; }

        __host__ __device__ void setSubTensor(int offsetRow, int offsetCol, Tensor const &mt) {
                return;
        }

        __host__ __device__ Tensor multiplyC(double scalar) {
                /// SmallTensor
                return Tensor(false);
        }

        __host__ __device__ Tensor plus(Tensor const &other) { return Tensor(false); }

        __host__ __device__ Tensor transpose() const { return Tensor(false); }

        
        /// cuBLAS -  column-major storage !
        __host__ __device__ static Tensor fromDeviceMem(double *dev_tensor, int rows, int cols) {
                Tensor t = Tensor(false);
                t.rows = rows;
                t.cols = cols;
                t.tensor_ref = dev_tensor;
                return t;
        }

      public:
        int rows = 0;
        int cols = 0;
        union {
                double tensor[4] = {};
                double *tensor_ref;  /// cuBLAS -  column-major storage !

        };
        bool nonMemOwning; /// cast_helper if true -> SmallTensor otherwise RefMatrixDouble
};


/// 2x2
class SmallTensor : public Tensor {
      public:
        __host__ __device__ SmallTensor() : Tensor(true) { 
            tensor[0] = 0.0;
            tensor[1] = 0.0;
            tensor[2] = 0.0;
            tensor[3] = 0.0;
        }

        __host__ __device__ SmallTensor(double value) : SmallTensor(value, value) {}

        __host__ __device__ SmallTensor(double a00, double a11) : SmallTensor(a00,0.0,0.0, a11) {}

        __host__ __device__ SmallTensor(double a00, double a01, double a10, double a11) : SmallTensor() {
                tensor[0] = a00;
                tensor[1] = a01;
                tensor[2] = a10;
                tensor[3] = a11;
        }               

        __host__ __device__ SmallTensor tensorR() {
                double a00 = 0.0;
                double a01 = -1.0;
                double a10 = 1.0;
                double a11 = 0.0;
                return SmallTensor(a00, a01, a10, a11);
        }        

        __host__ __device__ static SmallTensor diagonal(double diagonal) { return SmallTensor(diagonal); }
};


class Vector {
      public:
        __host__ __device__ Vector(){};

        __host__ __device__ 
        Vector(Vector const &other) {
                this->x = other.x;
                this->y = other.y;
        };

        __host__ __device__ Vector(double px, double py) : x(px), y(py) {}

        __host__ __device__
        Vector plus(Vector const &other) const { return Vector(this->x + other.x, this->y + other.y); }

        __host__ __device__
        Vector minus(Vector const &other) const { return Vector(this->x - other.x, this->y - other.y); }

        __host__ __device__
        double product(Vector const &other) const { return (this->x * other.x + this->y * other.y); }

        __host__ __device__
        Vector product(double scalar) const { return Vector(this->x * scalar, this->y * scalar); }

        __host__ __device__
        double cross(Vector const &other) const { return (this->x * other.y - this->y * other.x); }

        __host__ __device__
        Vector operator/(double scalar) const { return Vector(this->x / scalar, this->y / scalar); }

        __host__ __device__
        double length() const { return sqrt(this->x * this->x + this->y * this->y); }

        __host__ __device__
        Vector unit() const { return this->operator/(length()); }

        __host__ __device__
        Vector pivot() const { return Vector(-this->y, this->x); }

        __host__ __device__
        Vector Rot(double angle) {
                double rad = toRadians(angle);
                return Vector(this->x * cos(rad) - this->y * sin(rad), this->x * sin(rad) + this->y * cos(rad));
        }

        __host__ __device__
        SmallTensor cartesian(Vector const &rhs) {
                double a00 = this->x * rhs.x;
                double a01 = this->x * rhs.y;
                double a10 = this->y * rhs.x;
                double a11 = this->y * rhs.y;
                return SmallTensor(a00, a01, a10, a11);
        }

        __host__ __device__
        bool operator==(Vector const &other) const { return (this->x == other.x && this->y == other.y); }

      public:
        double x, y;
};

class Point : public Vector {
      public:
        Point() = default;

        Point(int id, double px, double py) : Vector(px, py), id(id) {}

      public:
        int id;
};

struct Geometric {
        Geometric(int id, int geometricTypeId, int p1, int p2, int p3, int a, int b, int c, int d)
            : id(id), geometricTypeId(geometricTypeId), p1(p1), p2(p2), p3(p3), a(a), b(b), c(c), d(d) {}

        int id;
        int geometricTypeId;
        int p1;
        int p2;
        int p3;
        int a;
        int b;
        int c;
        int d;
};

/**
 *   Geometric Object size in terms of point set size.
 */

struct Constraint {
        Constraint(int id, int constraintTypeId, int k, int l, int m, int n, int paramId, double vecX, double vecY)
            : id(id), constraintTypeId(constraintTypeId), k(k), l(l), m(m), n(n), paramId(paramId), vecX(vecX), vecY(vecY) {}

        int id;
        int constraintTypeId;
        int k;
        int l;
        int m;
        int n;
        int paramId;
        double vecX;
        double vecY;
};

struct Parameter {
        Parameter(int id, double value) : id(id), value(value) {}

        int id;
        double value;
};




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

#endif // _MODEL_CUH_