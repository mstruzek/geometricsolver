package com.mstruzek.msketch;

import com.mstruzek.controller.EventType;
import com.mstruzek.controller.Events;
import com.mstruzek.msketch.solver.GeometricSolver;
import com.mstruzek.msketch.solver.GeometricSolverImpl;
import com.mstruzek.msketch.solver.SolverStat;
import com.mstruzek.msketch.solver.StateReporter;

import java.util.Random;

import static com.mstruzek.controller.EventType.*;

/**
 * Klasa w ktorej przechowujemy caly model
 * matematyczny naszego szkicownika
 *
 * @author root
 */
public final class Model {


    private static GeometricSolver geometricSolver;

    private Model() {
        geometricSolver = new GeometricSolverImpl();
    }

    public static void solveSystem() {
        if (geometricSolver == null) {
            geometricSolver = new GeometricSolverImpl();
        }

        final StateReporter reporter = StateReporter.getInstance();

        SolverStat stat = new SolverStat();

        geometricSolver.solveSystem(stat);

        stat.report(reporter);

        Events.send(EventType.SOLVER_STAT_CHANGE, new Object[]{stat});
    }

    public static void addLine(int primitiveId, Vector v1, Vector v2) {
        add(new Line(primitiveId, v1, v2));
    }

    public static void addLine(Vector v1, Vector v2) {
        add(new Line(v1, v2));
    }

    public static void addCircle(int primitiveId, Vector v1, Vector v2) {
        add(new Circle(primitiveId, v1, v2));
    }

    public static void addCircle(Vector v1, Vector v2) {
        add(new Circle(v1, v2));
    }

    public static void addArc(int primitiveId, Vector v1, Vector v2, Vector v3) {
        add(new Arc(primitiveId, v1, v2, v3));
    }

    public static void addArc(Vector v1, Vector v2, Vector v3) {
        add(new Arc(v1, v2, v3));
    }

    public static void addPoint(int primitiveId, Vector v1) {
        add(new FreePoint(primitiveId, v1));
    }

    public static void addPoint(Vector v1) {
        add(new FreePoint(v1));
    }

    private static void add(GeometricPrimitive geometricPrimitive) {
        /// self registrations into GeometricPrimitive.dbPrimitives
        Events.send(PRIMITIVE_TABLE_INSERT, null);
    }

    public static void addConstraint(GeometricConstraintType constraintType, int K, int L, int M, int N, Double paramValue) {
        Point pK = null, pL = null, pM = null, pN = null;
        Parameter parameter = null;
        if (K < 0) {
            return;
        }
        pK = Point.dbPoint.get(K);
        pL = Point.dbPoint.get(L);
        pM = Point.dbPoint.get(M);
        pN = Point.dbPoint.get(N);

        if (paramValue != null && constraintType.isParametrized()) {
            parameter = new Parameter(paramValue);
        }

        addConstraint(Constraint.nextId(), constraintType, pK, pL, pM, pN, parameter);
    }

    public static void addConstraint(int constId, GeometricConstraintType constraintType, Point K, Point L, Point M, Point N, Parameter parameter) {
        switch (constraintType) {
            case Connect2Points:
                add(new ConstraintConnect2Points(constId, K, L));
                break;
            case HorizontalPoint:
                add(new ConstraintHorizontalPoint(constId, K, L));
                break;
            case VerticalPoint:
                add(new ConstraintVerticalPoint(constId, K, L));
                break;
            case FixPoint:
                add(new ConstraintFixPoint(constId, K));
                break;
            case ParametrizedXFix:
                add(new ConstraintParametrizedXFix(constId, K, parameter));
                add(parameter);
                break;
            case ParametrizedYFix:
                add(new ConstraintParametrizedYFix(constId, K, parameter));
                add(parameter);
                break;
            case LinesPerpendicular:
                add(new ConstraintLinesPerpendicular(constId, K, L, M, N));
                break;
            case LinesParallelism:
                add(new ConstraintLinesParallelism(constId, K, L, M, N));
                break;
            case Tangency:
//                add(new ConstraintTangency2(constId, K, L, M, N));
                add(new ConstraintTangency(constId, K, L, M, N));
                break;
            case Distance2Points:
                add(new ConstraintDistance2Points(constId, K, L, parameter));
                add(parameter);
                break;
            case Angle2Lines:
                add(new ConstraintAngle2Lines(constId, K, L, M, N, parameter));
                add(parameter);
                break;
            case DistancePointLine:
                add(new ConstraintDistancePointLine(constId, K, L, M, parameter));
                add(parameter);
                break;
            case EqualLength:
                add(new ConstraintEqualLength(constId, K, L, M, N));
                break;
            case ParametrizedLength:
                add(new ConstraintParametrizedLength(constId, K, L, M, N, parameter));
                add(parameter);
                break;
            case SetVertical:
                add(new ConstraintVertical(constId, K, L));
                break;
            case SetHorizontal:
                add(new ConstraintHorizontal(constId, K, L));
                break;
        }
    }

    public static void add(Constraint constraint) {
        /// self registration Constraint.dbConstraint
        Events.send(CONSTRAINT_TABLE_INSERT, new Object[]{});
    }

    public static void add(Parameter parameter) {
        /// self registration Parameter.dbParameter
        Events.send(PARAMETER_TABLE_INSERT, new Object[]{});
    }


    public static void evaluateGuidePoints() {
        for (Integer g : GeometricPrimitive.dbPrimitives.keySet()) {
            GeometricPrimitive.dbPrimitives.get(g).evaluateGuidePoints();
        }
    }

    public static void relaxControlPoints(double scale) {
        for (GeometricPrimitive geometricPrimitive : GeometricPrimitive.dbPrimitives.values()) {
            relaxPoint(geometricPrimitive.getP1(), scale);
            relaxPoint(geometricPrimitive.getP1(), scale);
            relaxPoint(geometricPrimitive.getP1(), scale);
        }
    }

    private static Random random = new Random();

    private static void relaxPoint(int pID, double scale) {
        if (pID == -1) return;
        Point point = Point.dbPoint.get(pID);
        double untenseX = point.getX() + scale * (random.nextDouble() - 0.5) * point.getX();
        double untenseY = point.getY() + scale * (random.nextDouble() - 0.5) * point.getY();
        point.setLocation(untenseX, untenseY);
    }
}
