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


__host__ __device__ double toRadians(double angdeg);



class Tensor {
      public:

        __host__ __device__ Tensor();

        __host__ __device__ Tensor(bool nonMemOwning);

        __host__ __device__ void setVector(int offsetRow, int offsetCol, Vector const& value);

        __host__ __device__ void plusSubTensor(int offsetRow, int offsetCol, Tensor const &mt);

        __host__ __device__ void setSubTensor(int offsetRow, int offsetCol, Tensor const &mt);

        __host__ __device__ Tensor multiplyC(double scalar);
        __host__ __device__ Tensor plus(Tensor const &other);

        __host__ __device__ Tensor transpose() const;

      public:
        int rows = 0;
        int cols = 0;
        bool nonMemOwning; /// cast_helper if true -> SmallTensor otherwise RefMatrixDouble
};

/// non-ownig reference wrapper
class TensorImpl : public Tensor {
      public:
        TensorImpl() : Tensor(false) {}

        double *tensor = nullptr; /// this memory is managed via cpu API into cuda runtime
};

/// 2x2
class SmallTensor : public Tensor {
      public:
        __host__ __device__ SmallTensor() : Tensor(true) {}

        __host__ __device__ SmallTensor(double a00, double a11);

        __host__ __device__ SmallTensor(double a00, double a01, double a10, double a11);

        __host__ __device__ static SmallTensor diagonal(double diagonal);
        __host__ __device__ static SmallTensor tensorR();

      public:
        double tensor[4] = {};
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

/// corespond to java implementations
struct SolverStat {
        SolverStat() = default;

        SolverStat(long startTime, long stopTime, long timeDelta, int size, int coefficientArity, int dimension, long accEvaluationTime, long accSolverTime,
                   bool convergence, double error, double constraintDelta, int iterations)
            : startTime(startTime), stopTime(stopTime), timeDelta(timeDelta), size(size), coefficientArity(coefficientArity), dimension(dimension),
              accEvaluationTime(accEvaluationTime), accSolverTime(accSolverTime), convergence(convergence), error(error), constraintDelta(constraintDelta),
              iterations(iterations) {}

        long startTime;
        long stopTime;
        long timeDelta;
        int size;
        int coefficientArity;
        int dimension;
        long accEvaluationTime;
        long accSolverTime;
        bool convergence;
        double error;
        double constraintDelta;
        int iterations;
};



} // namespace graph

#endif // _MODEL_CUH_