#ifndef _MODEL_H_
#define _MODEL_H_


struct Point
{
  Point(int id , double px, double py)
  : id(id), px(px), py(py) {}

  int id;
  double px;
  double py;
};


struct Geometric
{
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


struct Constraint
{
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


struct Parameter
{
  Parameter(int id, double value) : id(id), value(value)
   {}
  
  int id;
  double value;
};


/// corespond to java implementations
struct SolverStat
{
  
  SolverStat() = default;

  SolverStat(long startTime, long stopTime, long timeDelta, int size, int coefficientArity, int dimension, long accEvaluationTime, long accSolverTime, bool convergence, double error, double constraintDelta, int iterations)
      : startTime(startTime),
        stopTime(stopTime),
        timeDelta(timeDelta),
        size(size),
        coefficientArity(coefficientArity),
        dimension(dimension),
        accEvaluationTime(accEvaluationTime),
        accSolverTime(accSolverTime),
        convergence(convergence),
        error(error),
        constraintDelta(constraintDelta),
        iterations(iterations)
  {
  }

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



#endif // _MODEL_H_