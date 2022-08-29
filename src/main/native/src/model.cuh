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

#define __GPU_COMM_INL__ __forceinline__ __host__ __device__
#define __GPU_DEV_INL__ __forceinline__ __device__

class Vector;

enum StorageType
{
    row_major,   // row-major storage format
    column_major // column-major storage format - cuBlas, Fortran
};

///
/// Tensor memory u
///
class Layout
{
  public:
    StorageType mtype; /// column_major / row-major
    size_t ld;         /// leading dimension
    size_t rffset;     /// row offset
    size_t cffset;     /// column offset
};

/// obejmujemy dwa przypadki / jacobian i jacobian' transponowany

__GPU_COMM_INL__ static Layout defaultColumnMajor(size_t ld, size_t rffset, size_t cffset)
{
    return Layout{StorageType::column_major, ld, rffset, cffset};
}

__GPU_DEV_INL__ static Layout default2x2()
{
    __device__ static Layout u{StorageType::row_major, 2, 0, 0};
    return u;
}

__GPU_COMM_INL__ double getVectorX(Vector const &value);

__GPU_COMM_INL__ double getVectorY(Vector const &value);

class Tensor
{
  public:
    __GPU_COMM_INL__ Tensor(Layout u = default2x2()) : u(u)
    {
    }

    __GPU_COMM_INL__ Tensor(double a00, double a01, double a10, double a11) : Tensor()
    {
        tensor[0] = a00;
        tensor[1] = a01;
        tensor[2] = a10;
        tensor[3] = a11;
    }

    __GPU_COMM_INL__ void setVector(int offsetRow, int offsetCol, Vector const &value)
    {
        if (u.mtype == column_major)
        {
            if (colIntention)
            { /// vertical
                A[u.ld * (u.cffset + offsetCol) + u.rffset + offsetRow + 0] = getVectorX(value);
                A[u.ld * (u.cffset + offsetCol) + u.rffset + offsetRow + 1] = getVectorY(value);
            }
            else
            { /// horizontal
                A[u.ld * (u.cffset + offsetCol) + u.rffset + offsetRow] = getVectorX(value);
                A[u.ld * (u.cffset + offsetCol + 1) + u.rffset + offsetRow] = getVectorY(value);
            }
        }
    };

    __GPU_COMM_INL__ void setValue(int offsetRow, int offsetCol, double const &value)
    {
        if (u.mtype == column_major)
        {
            A[u.ld * (u.cffset + offsetCol) + u.rffset + offsetRow] = value;
        }
    }

    __GPU_COMM_INL__ double getValue(int offsetRow, int offsetCol) const
    {
        if (u.mtype == column_major)
        {
            return A[u.ld * (u.cffset + offsetCol) + u.rffset + offsetRow];
        }
        else if (u.mtype == row_major)
        {
            return tensor[2 * offsetRow + offsetCol];
        }
    }

    __GPU_COMM_INL__ void plusSubTensor(int offsetRow, int offsetCol, Tensor const &mt)
    {
        if (u.mtype == column_major && mt.u.mtype == row_major)
        {
            /// small tensor
            double a00 = mt.tensor[0];
            double a01 = mt.tensor[1];
            double a10 = mt.tensor[2];
            double a11 = mt.tensor[3];

            A[u.ld * (u.cffset + offsetCol) + u.rffset + offsetRow + 0] += a00;
            A[u.ld * (u.cffset + offsetCol) + u.rffset + offsetRow + 1] += a10;
            A[u.ld * (u.cffset + offsetCol + 1) + u.rffset + offsetRow + 0] += a01;
            A[u.ld * (u.cffset + offsetCol + 1) + u.rffset + offsetRow + 1] += a11;
        }
    }

    __GPU_COMM_INL__ void setSubTensor(int offsetRow, int offsetCol, Tensor const &mt)
    {
        if (u.mtype == column_major)
        {
            /// small tensor
            double a00 = mt.tensor[0];
            double a01 = mt.tensor[1];
            double a10 = mt.tensor[2];
            double a11 = mt.tensor[3];

            A[u.ld * (u.cffset + offsetCol) + u.rffset + offsetRow + 0] = a00;
            A[u.ld * (u.cffset + offsetCol) + u.rffset + offsetRow + 1] = a10;
            A[u.ld * (u.cffset + offsetCol + 1) + u.rffset + offsetRow + 0] = a01;
            A[u.ld * (u.cffset + offsetCol + 1) + u.rffset + offsetRow + 1] = a11;
        }
    }

    __GPU_COMM_INL__ Tensor multiplyC(double scalar)
    {
        if (u.mtype == row_major)
        {
            /// SmallTensor
            double a00 = tensor[0] * scalar;
            double a01 = tensor[1] * scalar;
            double a10 = tensor[2] * scalar;
            double a11 = tensor[3] * scalar;
            return Tensor(a00, a01, a10, a11);
        }
        return Tensor();
    }

    __GPU_COMM_INL__ Tensor plus(Tensor const &other)
    {
        if (u.mtype == row_major && other.u.mtype == row_major)
        {
            /// SmallTensor
            double a00 = tensor[0] + other.tensor[0];
            double a01 = tensor[1] + other.tensor[1];
            double a10 = tensor[2] + other.tensor[2];
            double a11 = tensor[3] + other.tensor[3];
            return Tensor(a00, a01, a10, a11);
        }
        return Tensor();
    }

    __GPU_COMM_INL__ Tensor transpose() const
    {
        double a00 = tensor[0];
        double a01 = tensor[1];
        double a10 = tensor[2];
        double a11 = tensor[3];
        return Tensor(a00, a10, a01, a11);
    }


  public:
    union {
        double tensor[4] = {0.0}; /// row-major storage - logical mistake !!!
        double *A;                /// parent reference tensor cuBLAS -  column-major storage !
    };
    Layout u;          /// memory u-layout  for parent  tensor A  - encodes offsets into view
    bool colIntention; /// vector operations
    int rows = 0;      /// range is not validated
    int cols = 0;      /// range is not validated
};


    /// cuBLAS -  column-major storage !
__GPU_COMM_INL__ static Tensor tensorDevMem(double *dev_tensor, Layout parent, int rows, int cols)
{
    Tensor t = Tensor(parent);
    t.A = dev_tensor;
    t.colIntention = rows > cols;
    t.rows = rows;
    t.cols = cols;
    return t;
}



#define DEGREES_TO_RADIANS 0.017453292519943295;

__GPU_COMM_INL__ double toRadians(double angdeg)
{
    return angdeg * DEGREES_TO_RADIANS;
}

/// 2x2
class SmallTensor
{
  public:
    __GPU_COMM_INL__ static Tensor tensorR()
    {
        double a00 = 0.0;
        double a01 = -1.0;
        double a10 = 1.0;
        double a11 = 0.0;
        return Tensor(a00, a01, a10, a11);
    }

    __GPU_COMM_INL__ static Tensor rotation(double alfa)
    {
        double radians = toRadians(alfa);
        double a00 = cos(radians);
        double a01 = -1.0 * sin(radians);
        double a10 = sin(radians);
        double a11 = cos(radians);
        return Tensor(a00, a01, a10, a11);
    }

    __GPU_COMM_INL__ static Tensor identity(double value)
    {
        return Tensor(value, 0.0, 0.0, value);
    }

    __GPU_COMM_INL__ static Tensor diagonal(double diagonal)
    {
        return Tensor(diagonal, 0.0, 0.0, diagonal);
    }
};

///
/// Transpose Tensor Adapter - utility class for transposed storage operations into output tensor
///  -- transpose/inversed offsetRow/offsetCol order
/// 
class AdapterTensor : public Tensor
{

  public:
    __GPU_COMM_INL__ AdapterTensor(Layout layout) : Tensor(layout)
    {
    }

    __GPU_COMM_INL__ void setVector(int offsetRow, int offsetCol, Vector const &value)
    {
        Tensor::setVector(offsetCol, offsetRow, value);
    }

    __GPU_COMM_INL__ void setValue(int offsetRow, int offsetCol, double const &value)
    {
        Tensor::setValue(offsetCol, offsetRow, value);
    }

    __GPU_COMM_INL__ void setSubTensor(int offsetRow, int offsetCol, Tensor const &mt)
    {
        Tensor::plusSubTensor(offsetCol, offsetRow, mt.transpose());
    }
};

// transponsed operations 
__GPU_COMM_INL__ static AdapterTensor transposeTensorDevMem(double *dev_tensor, Layout parent, int rows, int cols)
{
    AdapterTensor t = AdapterTensor(parent);
    t.A = dev_tensor;
    t.colIntention = rows > cols;
    t.rows = rows;
    t.cols = cols;
    return t;
}



class Vector
{
  public:
    __GPU_COMM_INL__ Vector(){};

    __GPU_COMM_INL__ Vector(Vector const &other)
    {
        this->x = other.x;
        this->y = other.y;
    };

    __GPU_COMM_INL__ Vector(double px, double py) : x(px), y(py)
    {
    }

    __GPU_COMM_INL__ Vector plus(Vector const &other) const
    {
        return Vector(this->x + other.x, this->y + other.y);
    }

    __GPU_COMM_INL__ Vector minus(Vector const &other) const
    {
        return Vector(this->x - other.x, this->y - other.y);
    }

    __GPU_COMM_INL__ double product(Vector const &other) const
    {
        return (this->x * other.x + this->y * other.y);
    }

    __GPU_COMM_INL__ Vector product(double scalar) const
    {
        return Vector(this->x * scalar, this->y * scalar);
    }

    __GPU_COMM_INL__ double cross(Vector const &other) const
    {
        return (this->x * other.y - this->y * other.x);
    }

    __GPU_COMM_INL__ Vector operator/(double scalar) const
    {
        return Vector(this->x / scalar, this->y / scalar);
    }

    __GPU_COMM_INL__ double length() const
    {
        return sqrt(this->x * this->x + this->y * this->y);
    }

    __GPU_COMM_INL__ Vector unit() const
    {
        return this->operator/(length());
    }

    __GPU_COMM_INL__ Vector pivot() const
    {
        return Vector(-this->y, this->x);
    }

    __GPU_COMM_INL__ Vector Rot(double angle)
    {
        double rad = toRadians(angle);
        return Vector(this->x * cos(rad) - this->y * sin(rad), this->x * sin(rad) + this->y * cos(rad));
    }

    __GPU_COMM_INL__ Tensor cartesian(Vector const &rhs)
    {
        double a00 = this->x * rhs.x;
        double a01 = this->x * rhs.y;
        double a10 = this->y * rhs.x;
        double a11 = this->y * rhs.y;
        return Tensor(a00, a01, a10, a11);
    }

    __GPU_COMM_INL__ bool operator==(Vector const &other) const
    {
        return (this->x == other.x && this->y == other.y);
    }

  public:
    double x, y;
};

typedef Vector PPoint;

__GPU_COMM_INL__ double getVectorX(Vector const &value)
{
    return value.x;
}

__GPU_COMM_INL__ double getVectorY(Vector const &value)
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