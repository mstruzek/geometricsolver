#ifndef _MODEL_H_
#define _MODEL_H_

#include <cuda_runtime_api.h>
#include <math.h>

namespace stage {

//// odseparowac cuh i cu file

class MatrixDouble {
      public:
        __device__ MatrixDouble(bool typeNonReference) : typeNonReference(typeNonReference) {}

        __device__ void setVector(int offsetRow, int offsetCol, Vector const &value) {

                /// ALL - dwa przypadki na miejscu
        }

        __device__ void plusSubMatrix(int offsetRow, int offsetCol, MatrixDouble const &mt) {

                /// ALL -  dwa przypadki na miejscu
        }

        __device__ void setSubMatrix(int offsetRow, int offsetCol, MatrixDouble const &mt) {

                /// ALL - dwa przypadki na miejscu
        }

        __device__ MatrixDouble multiplyC(double scalar){

            /// SmallMatrixDouble
        };

        __device__ MatrixDouble plus(MatrixDouble const &other) {

                /// SmallMatrixDouble
        }

        int rows;
        int cols;
        bool typeNonReference; /// cast_helper if true -> SmallMatrixDouble otherwise RefMatrixDouble
};

/// non-ownig reference wrapper
class RefMatrixDouble : public MatrixDouble {
      public:
        RefMatrixDouble() : MatrixDouble(false) {}
        double *tensor; /// this memory is managed via cpu API into cuda runtime
};

/// 2x2
class SmallMatrixDouble : public MatrixDouble {
      public:
        __device__ SmallMatrixDouble() : MatrixDouble(true) {}

        __device__ SmallMatrixDouble(double a00, double a11) : SmallMatrixDouble() {
                tensor[0] = a00;
                tensor[3] = a11;
        }

        __device__ SmallMatrixDouble(double a00, double a01, double a10, double a11): SmallMatrixDouble() {
                tensor[0] = a00;
                tensor[1] = a01;
                tensor[2] = a10;
                tensor[3] = a11;
        }

        __device__ static SmallMatrixDouble diagonal(double diagonal) { return SmallMatrixDouble(diagonal, diagonal); }
        __device__ static SmallMatrixDouble matrixR() {
                double a00 = 0.0;
                double a01 = -1.0;
                double a10 = 1.0;
                double a11 = 0.0;
                return SmallMatrixDouble(a00, a01, a10, a11);
        }

      public:
        double tensor[4];
};

#define DEGREES_TO_RADIANS 0.017453292519943295;

__device__ double toRadians(double angdeg) { return angdeg * DEGREES_TO_RADIANS; }

class Vector {
      public:
        __device__ Vector() = default;

        __device__ Vector(Vector const &other) = default;

        __device__ Vector(double px, double py) : x(px), y(py) {}

        __device__ Vector plus(Vector const &other) const { return Vector(this->x + other.x, this->y + other.y); }

        __device__ Vector minus(Vector const &other) const { return Vector(this->x - other.x, this->y - other.y); }

        __device__ double product(Vector const &other) const { return (this->x * other.x + this->y * other.y); }

        __device__ Vector product(double scalar) const { return Vector(this->x * scalar, this->y * scalar); }

        __device__ double cross(Vector const &other) const { return (this->x * other.y - this->y * other.x); }

        __device__ Vector operator/(double scalar) const { return Vector(this->x / scalar, this->y / scalar); }

        __device__ double length() const { return sqrt(this->x * this->x + this->y * this->y); }

        __device__ Vector unit() const { return this->operator/(length()); }

        __device__ Vector pivot() const { return Vector(-this->y, this->x); }

        __device__ Vector Rot(double angle) {
                double rad = toRadians(angle);
                return Vector(this->x * cos(rad) - this->y * sin(rad), this->x * sin(rad) + this->y * cos(rad));
        }

        SmallMatrixDouble cartesian(Vector const &rhs) {
                double a00 = this->x * rhs.x;
                double a01 = this->x * rhs.y;
                double a10 = this->y * rhs.x;
                double a11 = this->y * rhs.y;
                return SmallMatrixDouble(a00, a01, a10, a11);
        }

        __device__ bool operator==(Vector const &other) const { return (this->x == other.x && this->y == other.y); }

      public:
        double x, y;
};

class Point : public Vector {
      public:
        Point() = default;

        Point(int id, double px, double py) : Vector(px, py), id(id) {}

        Point(Point const &other) = default;
        Point(Point &&other) = default;

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

struct Constraint {
        Constraint(int id, int constrajintTypeId, int k, int l, int m, int n, int paramId, double vecX, double vecY)
            : id(id), constrajintTypeId(constrajintTypeId), k(k), l(l), m(m), n(n), paramId(paramId), vecX(vecX), vecY(vecY) {}

        int id;
        int constrajintTypeId;
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

} // namespace stage

#endif // _MODEL_H_