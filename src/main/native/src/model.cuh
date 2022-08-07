#ifndef _MODEL_H_
#define _MODEL_H_

#include <cuda_runtime_api.h>
#include <math.h>

namespace graph {

//// odseparowac cuh i cu file


class Vector;


class Tensor {
      public:
        Tensor() : nonMemOwning(true) {}

        Tensor(bool nonMemOwning) : nonMemOwning(nonMemOwning) {}

        void setVector(int offsetRow, int offsetCol, Vector const &value) {

                /// ALL - dwa przypadki na miejscu
        }

        void plusSubTensor(int offsetRow, int offsetCol, Tensor const &mt) {

                /// ALL -  dwa przypadki na miejscu
        }

        void setSubTensor(int offsetRow, int offsetCol, Tensor const &mt) {

                /// ALL - dwa przypadki na miejscu
        }

        Tensor multiplyC(double scalar){

            /// SmallTensor
        };

        Tensor plus(Tensor const &other) {

                /// SmallTensor
        }


        Tensor transpose() const {
                
        }

        int rows;
        int cols;
        bool nonMemOwning; /// cast_helper if true -> SmallTensor otherwise RefMatrixDouble
};

/// non-ownig reference wrapper
class TensorImpl : public Tensor {
      public:
        TensorImpl() : Tensor(false) {}

        double *tensor; /// this memory is managed via cpu API into cuda runtime
};

/// 2x2
class SmallTensor : public Tensor {
      public:
        SmallTensor() : Tensor(true) {}

        SmallTensor(double a00, double a11) : SmallTensor() {
                tensor[0] = a00;
                tensor[3] = a11;
        }

        SmallTensor(double a00, double a01, double a10, double a11) : SmallTensor() {
                tensor[0] = a00;
                tensor[1] = a01;
                tensor[2] = a10;
                tensor[3] = a11;
        }

        static SmallTensor diagonal(double diagonal) { return SmallTensor(diagonal, diagonal); }
        static SmallTensor tensorR() {
                double a00 = 0.0;
                double a01 = -1.0;
                double a10 = 1.0;
                double a11 = 0.0;
                return SmallTensor(a00, a01, a10, a11);
        }

      public:
        double tensor[4];
};

#define DEGREES_TO_RADIANS 0.017453292519943295;

double toRadians(double angdeg) { return angdeg * DEGREES_TO_RADIANS; }

class Vector {
      public:
        Vector() = default;

        Vector(Vector const &other) = default;

        Vector(double px, double py) : x(px), y(py) {}

        Vector plus(Vector const &other) const { return Vector(this->x + other.x, this->y + other.y); }

        Vector minus(Vector const &other) const { return Vector(this->x - other.x, this->y - other.y); }

        double product(Vector const &other) const { return (this->x * other.x + this->y * other.y); }

        Vector product(double scalar) const { return Vector(this->x * scalar, this->y * scalar); }

        double cross(Vector const &other) const { return (this->x * other.y - this->y * other.x); }

        Vector operator/(double scalar) const { return Vector(this->x / scalar, this->y / scalar); }

        double length() const { return sqrt(this->x * this->x + this->y * this->y); }

        Vector unit() const { return this->operator/(length()); }

        Vector pivot() const { return Vector(-this->y, this->x); }

        Vector Rot(double angle) {
                double rad = toRadians(angle);
                return Vector(this->x * cos(rad) - this->y * sin(rad), this->x * sin(rad) + this->y * cos(rad));
        }

        SmallTensor cartesian(Vector const &rhs) {
                double a00 = this->x * rhs.x;
                double a01 = this->x * rhs.y;
                double a10 = this->y * rhs.x;
                double a11 = this->y * rhs.y;
                return SmallTensor(a00, a01, a10, a11);
        }

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
int geometricSetSize(graph::Geometric const& geometric) {
        switch (geometric.geometricTypeId) {
        case GEOMETRIC_TYPE_ID_FreePoint:
                return 3 * 2;
        case GEOMETRIC_TYPE_ID_Line:
                return 4 * 2;
        case GEOMETRIC_TYPE_ID_Circle:
                return 4 * 2;
        case GEOMETRIC_TYPE_ID_Arc:
                return 7 * 2;
        }
}

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

int constraintSize(graph::Constraint const& constraint) {
        switch (constraint.constraintTypeId) {
        case CONSTRAINT_TYPE_ID_FixPoint:
                return 2;
        case CONSTRAINT_TYPE_ID_ParametrizedXFix:
                return 1;
        case CONSTRAINT_TYPE_ID_ParametrizedYFix:
                return 1;
        case CONSTRAINT_TYPE_ID_Connect2Points:
                return 2;
        case CONSTRAINT_TYPE_ID_HorizontalPoint:
                return 1;
        case CONSTRAINT_TYPE_ID_VerticalPoint:
                return 1;
        case CONSTRAINT_TYPE_ID_LinesParallelism:
                return 1;
        case CONSTRAINT_TYPE_ID_LinesPerpendicular:
                return 1;
        case CONSTRAINT_TYPE_ID_EqualLength:
                return 1;
        case CONSTRAINT_TYPE_ID_ParametrizedLength:
                return 1;
        case CONSTRAINT_TYPE_ID_Tangency:
                return 1;
        case CONSTRAINT_TYPE_ID_CircleTangency:
                return 1;
        case CONSTRAINT_TYPE_ID_Distance2Points:
                return 1;
        case CONSTRAINT_TYPE_ID_DistancePointLine:
                return 1;
        case CONSTRAINT_TYPE_ID_Angle2Lines:
                return 1;
        case CONSTRAINT_TYPE_ID_SetHorizontal:
                return 1;
        case CONSTRAINT_TYPE_ID_SetVertical:
                return 1;
        }
}

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

#endif // _MODEL_H_