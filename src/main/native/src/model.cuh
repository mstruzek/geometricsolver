#ifndef _MODEL_H_
#define _MODEL_H_

#include "cuda_runtime.h"
#include "math.h"

//#//include <cuda/std/detail/libcxx/include/cstdlib> 


#include <stdio.h>

#include "model_config.h"

namespace graph
{


struct Geometric;

struct Constraint;

int constraintSize(Constraint const &constraint);
int geometricSetSize(Geometric const &geometric);

class Vector;

__forceinline__ __device__ __host__ double getVectorX(Vector const &value);

__forceinline__ __device__ __host__ double getVectorY(Vector const &value);

class Tensor
{
  public:
    __forceinline__ __host__ __device__ Tensor(bool nonMemOwning = true) : nonMemOwning(nonMemOwning)
    {
    }

    __forceinline__ __host__ __device__ Tensor(double a00, double a01, double a10, double a11) : Tensor(true)
    {
        tensor[0] = a00;
        tensor[1] = a01;
        tensor[2] = a10;
        tensor[3] = a11;
    }

    __forceinline__ __host__ __device__ __forceinline__ void Tensor::setVector(int offsetRow, int offsetCol,
                                                                              Vector const &value)
    {
        if (tensor_ref != nullptr)
        {
            tensor_ref[ld * offsetCol + offsetRow + 0] = getVectorX(value);
            tensor_ref[ld * offsetCol + offsetRow + 1] = getVectorY(value);
        }
    };

    __forceinline__ __host__ __device__ void setValue(int offsetRow, int offsetCol, double const &value)
    {
        if (tensor_ref != nullptr)
        {
            tensor_ref[ld * offsetCol + offsetRow] = value;
        }
    }

    __host__ __device__ void plusSubTensor(int offsetRow, int offsetCol, Tensor const &mt)
    {
        if (tensor_ref != nullptr && mt.nonMemOwning == true)
        {
            /// small tensor
            double a00 = mt.tensor[0];
            double a01 = mt.tensor[1];
            double a10 = mt.tensor[2];
            double a11 = mt.tensor[3];

            tensor_ref[ld * offsetCol + offsetRow + 0] += a00;
            tensor_ref[ld * offsetCol + offsetRow + 1] += a10;
            tensor_ref[ld * (offsetCol + 1) + offsetRow + 0] += a01;
            tensor_ref[ld * (offsetCol + 1) + offsetRow + 1] += a11;
        }
    }

    __forceinline__ __host__ __device__ void setSubTensor(int row, int offsetCol, Tensor const &mt)
    {
        if (tensor_ref != nullptr && mt.nonMemOwning == true)
        {
            /// small tensor
            double a00 = mt.tensor[0];
            double a01 = mt.tensor[1];
            double a10 = mt.tensor[2];
            double a11 = mt.tensor[3];

            tensor_ref[ld * offsetCol + row + 0] = a00;
            tensor_ref[ld * offsetCol + row + 1] = a10;
            tensor_ref[ld * (offsetCol + 1) + row + 0] = a01;
            tensor_ref[ld * (offsetCol + 1) + row + 1] = a11;
        }
    }

    __forceinline__ __host__ __device__ Tensor multiplyC(double scalar)
    {
        if (nonMemOwning == true)
        {
            /// SmallTensor
            double a00 = tensor[0] * scalar;
            double a01 = tensor[1] * scalar;
            double a10 = tensor[2] * scalar;
            double a11 = tensor[3] * scalar;
            return Tensor(a00, a01, a10, a11);
        }               
        return Tensor(false);
    }

    __forceinline__ __host__ __device__ Tensor plus(Tensor const &other)
    {
        if (nonMemOwning == true && other.nonMemOwning == true)
        {
            /// SmallTensor
            double a00 = tensor[0] + other.tensor[0];
            double a01 = tensor[1] + other.tensor[1];
            double a10 = tensor[2] + other.tensor[2];
            double a11 = tensor[3] + other.tensor[3];
            return Tensor(a00, a01, a10, a11);
        }        
        return Tensor(false);
    }

    /// cuBLAS -  column-major storage !
    __forceinline__ __host__ __device__ static Tensor fromDeviceMem(double *dev_tensor, int ld, int cols)
    {
        Tensor t = Tensor(false);
        t.ld = ld;
        t.cols = cols;
        t.tensor_ref = dev_tensor;
        return t;
    }

  public:
    int ld = 0;
    int cols = 0;
    union {
        double tensor[4] = {0.0};
        double *tensor_ref; /// cuBLAS -  column-major storage !
    };
    bool nonMemOwning; /// cast_helper if true -> SmallTensor otherwise RefMatrixDouble
};

#define DEGREES_TO_RADIANS 0.017453292519943295;

__host__ __device__ double toRadians(double angdeg)
{
    return angdeg * DEGREES_TO_RADIANS;
}

/// 2x2
class SmallTensor
{
  public:
    __forceinline__ __host__ __device__ static Tensor tensorR()
    {
        double a00 = 0.0;
        double a01 = -1.0;
        double a10 = 1.0;
        double a11 = 0.0;
        return Tensor(a00, a01, a10, a11);
    }

    __forceinline__ __host__ __device__ static Tensor rotation(double alfa)
    {
        double radians = toRadians(alfa);
        double a00 = cos(radians);
        double a01 = -1.0 * sin(radians);
        double a10 = sin(radians);
        double a11 = cos(radians);
        return Tensor(a00, a01, a10, a11);
    }

    __forceinline__ __host__ __device__ static Tensor identity(double value)
    {
        return Tensor(value, 0.0, 0.0, value);
    }

    __forceinline__ __host__ __device__ static Tensor diagonal(double diagonal)
    {
        return Tensor(diagonal, 0.0, 0.0, diagonal);
    }
};

class Vector
{
  public:
    __host__ __device__ Vector(){};

    __host__ __device__ Vector(Vector const &other)
    {
        this->x = other.x;
        this->y = other.y;
    };

    __host__ __device__ Vector(double px, double py) : x(px), y(py)
    {
    }

    __host__ __device__ Vector plus(Vector const &other) const
    {
        return Vector(this->x + other.x, this->y + other.y);
    }

    __host__ __device__ Vector minus(Vector const &other) const
    {
        return Vector(this->x - other.x, this->y - other.y);
    }

    __host__ __device__ double product(Vector const &other) const
    {
        return (this->x * other.x + this->y * other.y);
    }

    __host__ __device__ Vector product(double scalar) const
    {
        return Vector(this->x * scalar, this->y * scalar);
    }

    __host__ __device__ double cross(Vector const &other) const
    {
        return (this->x * other.y - this->y * other.x);
    }

    __host__ __device__ Vector operator/(double scalar) const
    {
        return Vector(this->x / scalar, this->y / scalar);
    }

    __host__ __device__ __forceinline__ double length() const
    {
        return sqrt(this->x * this->x + this->y * this->y);
    }

    __host__ __device__ __forceinline__ Vector unit() const
    {
        return this->operator/(length());
    }

    __host__ __device__ __forceinline__ Vector pivot() const
    {
        return Vector(-this->y, this->x);
    }

    __host__ __device__ __forceinline__ Vector Rot(double angle)
    {
        double rad = toRadians(angle);
        return Vector(this->x * cos(rad) - this->y * sin(rad), this->x * sin(rad) + this->y * cos(rad));
    }

    __host__ __device__ __forceinline__ Tensor cartesian(Vector const &rhs)
    {
        double a00 = this->x * rhs.x;
        double a01 = this->x * rhs.y;
        double a10 = this->y * rhs.x;
        double a11 = this->y * rhs.y;
        return Tensor(a00, a01, a10, a11);
    }

    __host__ __device__ __forceinline__ bool operator==(Vector const &other) const
    {
        return (this->x == other.x && this->y == other.y);
    }

  public:
    double x, y;
};


typedef Vector PPoint;

__forceinline__ __device__ __host__ double getVectorX(Vector const &value)
{
    return value.x;
}

__forceinline__ __device__ __host__ double getVectorY(Vector const &value)
{
    return value.y;
}

class Point : public Vector
{
  public:
    Point() = default;

    Point(int id, double px, double py) : Vector(px, py), id(id)
    {
    }

  public:
    int id;
};

struct Geometric
{
    Geometric(int id, int geometricTypeId, int p1, int p2, int p3, int a, int b, int c, int d)
        : id(id), geometricTypeId(geometricTypeId), p1(p1), p2(p2), p3(p3), a(a), b(b), c(c), d(d)
    {
    }

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

struct Constraint
{
    Constraint(int id, int constraintTypeId, int k, int l, int m, int n, int paramId, double vecX, double vecY)
        : id(id), constraintTypeId(constraintTypeId), k(k), l(l), m(m), n(n), paramId(paramId), vecX(vecX), vecY(vecY)
    {
    }

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

struct Parameter
{
    Parameter(int id, double value) : id(id), value(value)
    {
    }

    int id;
    double value;
};

int constraintSize(Constraint const &constraint)
{
    switch (constraint.constraintTypeId)
    {
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

int geometricSetSize(Geometric const &geometric)
{
    switch (geometric.geometricTypeId)
    {
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

} // namespace graph

#endif // _MODEL_CUH_