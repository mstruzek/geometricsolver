#ifndef _MODEL_H_
#define _MODEL_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"

//#//include <cuda/std/detail/libcxx/include/cstdlib>

#include <stdio.h>

#include "model_config.h"

#include "tensor_layout.cuh"

#include "version.cuh"

#ifndef __GPU_COMM_INL__
#define __GPU_COMM_INL__ __host__ __device__
#endif

#ifndef __GPU_DEV_INL__
#define __GPU_DEV_INL__ __device__
#endif

#ifndef __GPU_DEV_INLF__
#define __GPU_DEV_INLF__ __forceinline__ __device__
#endif

namespace graph {

/// ================================================================================///
/// ================================================================================///
/// ================================================================================///
/// ================================================================================///

class BlockLayout;

template <typename> 
class Tensor;

//=================================================================================

class Vector {
  public:

#if !defined(__NVCC__) 
        Vector(double px, double py) : x(px), y(py) {}
#endif 

#ifdef __NVCC__

    __GPU_DEV_INLF__ Vector() : x(0.0), y(0.0){};

    __GPU_DEV_INLF__ Vector(Vector const &other) {
        this->x = other.x;
        this->y = other.y;
    };

    __GPU_DEV_INLF__ Vector(double px, double py) : x(px), y(py) {}

    __GPU_DEV_INLF__ Vector plus(Vector const &other) const { return Vector(this->x + other.x, this->y + other.y); }

    __GPU_DEV_INLF__ Vector minus(Vector const &other) const { return Vector(this->x - other.x, this->y - other.y); }

    __GPU_DEV_INLF__ double product(Vector const &other) const { return (this->x * other.x + this->y * other.y); }

    __GPU_DEV_INLF__ Vector product(double scalar) const { return Vector(this->x * scalar, this->y * scalar); }

    __GPU_DEV_INLF__ double cross(Vector const &other) const { return (this->x * other.y - this->y * other.x); }

    __GPU_DEV_INLF__ Vector operator/(double scalar) const { return Vector(this->x / scalar, this->y / scalar); }

    __GPU_DEV_INLF__ double length() const { return sqrt(this->x * this->x + this->y * this->y); }

    __GPU_DEV_INLF__ Vector unit() const { return this->operator/(length()); }

    __GPU_DEV_INLF__ Vector pivot() const { return Vector(-this->y, this->x); }

    __GPU_DEV_INL__ Vector Rot(double angle);

    __GPU_DEV_INL__ Tensor<BlockLayout> cartesian(Vector const &rhs);

    __GPU_DEV_INLF__ bool operator==(Vector const &other) const { return (this->x == other.x && this->y == other.y); }

#endif // __NVCC__

  public:
    double x, y;
};

typedef Vector PPoint;

#ifdef __NVCC__

__GPU_DEV_INLF__ double getVectorX(Vector const &value) { return value.x; }

__GPU_DEV_INLF__ double getVectorY(Vector const &value) { return value.y; }

#endif  // __NVCC__

/// ================================================================================///
/// ================================================================================///
/// ================================================================================///
/// ================================================================================///

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
        : id(id), constraintTypeId(constraintTypeId), k(k), l(l), m(m), n(n), paramId(paramId), vecX(vecX), vecY(vecY) {
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

struct Parameter {
    Parameter(int id, double value) : id(id), value(value) {}

    int id;
    double value;
};



/// Evaluate constraint leading dimension
int constraintSize(Constraint const &constraint);

/// Evaluate geometric object leading dimension  [ ld x ld ]
int geometricSetSize(Geometric const &geometric);

/// map computation mode from computation id
ComputationMode getComputationMode(int computationId);

/// accWriteCooStiffTensor
int tensorOpsCooStiffnesCoefficients(Geometric const &geometric);

/// accWriteCooJacobianTensor
int tensorOpsCooConstraintJacobian(Constraint const &constraint);


} // namespace graph


//=================================================================================

#undef NVECTOR_DEBUG

template <typename TObject> class NVector {

  public:
    __GPU_COMM_INL__ NVector() : _data(NULL), _size(0) {}
    __GPU_COMM_INL__ NVector(TObject *data, size_t size) : _data(data), _size(size) {}

    __GPU_COMM_INL__ NVector(const NVector &right) { *this = right; };

    __GPU_COMM_INL__ NVector &operator=(const NVector<TObject> &right) {
        _data = right._data;
        _size = right._size;
        return *this;
    };

    __GPU_COMM_INL__ TObject &operator[](const size_t idx) const {
#ifdef NVECTOR_DEBUG
        if (idx >= _size)
            printf("illegal memmory access : mem (%p)  , id ( %zu )\n", _data, idx);
#endif
        return _data[idx];
    }

  private:
    size_t _size;
    TObject *_data;
};


namespace quda {

    template <typename> struct printer;

    template <> struct printer<graph::Point> {
    __device__ __host__ void operator()(int i, const graph::Point& object) { 
        printf("%d  ( %d )[ %f , %f ]\n", i, object.id, object.x, object.y); 
    }
};
}

//=================================================================================

#endif // _MODEL_CUH_