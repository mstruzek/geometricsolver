package com.mstruzek.msketch;

import com.mstruzek.msketch.solver.GeometricSolver;
import com.mstruzek.msketch.solver.GeometricSolverImpl;
import com.mstruzek.msketch.solver.SolverStat;
import com.mstruzek.msketch.solver.StateReporter;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import static java.lang.System.out;

public class ConstraintConvergenceTest {

    GeometricSolver geometricSolver;
    SolverStat solverStat;
    StateReporter reporter;

    @Before
    public void beforeTest() {
        geometricSolver = new GeometricSolverImpl();
        reporter = StateReporter.getInstance();
        solverStat = new SolverStat();
    }

    @After
    public void afterTest() {
        solverStat.report(reporter);

        Constraint.dbConstraint.clear();
        Parameter.dbParameter.clear();
        GeometricPrimitive.dbPrimitives.clear();
        Point.dbPoint.clear();

        Constraint.constraintCounter = 0;
        Parameter.parameterCounter = 0;
        GeometricPrimitive.primitiveCounter = 0;
        Point.pointCounter = 0;
    }

    @Test
    public void convergenceConstraintFixPoint() {
        Point p10 = new Point(1, 0.0, 40.0);

        FreePoint f10 = new FreePoint(p10);
        f10.setAssociateConstraints(null);

        Constraint constraint = new ConstraintFixPoint(Constraint.nextId(), p10);
        p10.setLocation(100.0, 100.0);

        geometricSolver.solveSystem(solverStat);

        Assert.assertEquals(0, solverStat.iterations);
        Assert.assertTrue(solverStat.convergence);
        Assert.assertTrue(solverStat.delta < 10e-10);
        Assert.assertTrue(solverStat.constraintDelta < 10e-10);
        Assert.assertTrue(constraint.getNorm() < 10e-10);
    }

    @Test
    public void convergenceConstraintParametrizedXFix() {
        Point p10 = new Point(1, 40.0, 40.0);
        FreePoint f10 = new FreePoint(p10);
        f10.setAssociateConstraints(null);
        Parameter parameter = new Parameter(120.0); // fixed X coordinate
        Constraint constraint = new ConstraintParametrizedXFix(Constraint.nextId(), p10, parameter);

        geometricSolver.solveSystem(solverStat);

        Assert.assertEquals(0, solverStat.iterations);
        Assert.assertTrue(solverStat.convergence);
        Assert.assertTrue(solverStat.delta < 10e-10);
        Assert.assertTrue(solverStat.constraintDelta < 10e-10);
        Assert.assertTrue(constraint.getNorm() < 10e-10);
        Assert.assertEquals(p10.getX(), parameter.getValue(), 10e-10);
    }

    @Test
    public void convergenceConstraintParametrizedYFix() {
        Point p10 = new Point(1, 40.0, 40.0);
        FreePoint f10 = new FreePoint(p10);
        f10.setAssociateConstraints(null);
        Parameter parameter = new Parameter(120.0); // fixed Y coordinate
        Constraint constraint = new ConstraintParametrizedYFix(Constraint.nextId(), p10, parameter);

        geometricSolver.solveSystem(solverStat);

        Assert.assertEquals(0, solverStat.iterations);
        Assert.assertTrue(solverStat.convergence);
        Assert.assertTrue(solverStat.delta < 10e-10);
        Assert.assertTrue(solverStat.constraintDelta < 10e-10);
        Assert.assertTrue(constraint.getNorm() < 10e-10);
        Assert.assertEquals(p10.getY(), parameter.getValue(), 10e-10);
    }

    @Test
    public void convergenceConstraintConnect2Points() {
        Point p10 = new Point(1, 0.0, 40.0);
        Point p20 = new Point(4, 40.0, 0.0);

        FreePoint f10 = new FreePoint(p10);
        FreePoint f20 = new FreePoint(p20);
        f10.setAssociateConstraints(null);
        f20.setAssociateConstraints(null);

        Constraint constraint = new ConstraintConnect2Points(Constraint.nextId(), p10, p20);

        geometricSolver.solveSystem(solverStat);

        Assert.assertEquals(0, solverStat.iterations);
        Assert.assertTrue(solverStat.convergence);
        Assert.assertTrue(solverStat.delta < 10e-10);
        Assert.assertTrue(solverStat.constraintDelta < 10e-10);
        Assert.assertTrue(constraint.getNorm() < 10e-10);
    }

    @Test
    public void convergenceConstraintLinesPerpendicular() {
        Point p10 = new Point(1, 0.0, 40.0);
        Point p20 = new Point(2, 10.0, 80.0);
        Point p30 = new Point(5, 40.0, 0.0);
        Point p40 = new Point(6, 90.0, 10.0);

        Line f10 = new Line(p10, p20);
        Line f20 = new Line(p30, p40);
        f10.setAssociateConstraints(null);
        f20.setAssociateConstraints(null);

        Constraint constraint = new ConstraintLinesPerpendicular(Constraint.nextId(), p10, p20, p30, p40);

        geometricSolver.solveSystem(solverStat);

        Assert.assertEquals(2, solverStat.iterations);
        Assert.assertTrue(solverStat.convergence);
        Assert.assertTrue(solverStat.delta < 10e-2);
        Assert.assertTrue(solverStat.constraintDelta < 10e-5);
        Assert.assertTrue(constraint.getNorm() < 10e-5);
    }

    @Test
    public void convergenceConstraintHorizontal() {
        Point p10 = new Point(1, 0.0, 0.0);
        Point p20 = new Point(2, 40.0, 40.0);

        Line f10 = new Line(p10, p20);
        f10.setAssociateConstraints(null);

        Constraint constraint = new ConstraintHorizontal(Constraint.nextId(), p10, p20);

        geometricSolver.solveSystem(solverStat);

        Assert.assertEquals(0, solverStat.iterations);
        Assert.assertTrue(solverStat.convergence);
        Assert.assertTrue(solverStat.delta < 10e-2);
        Assert.assertTrue(solverStat.constraintDelta < 10e-5);
        Assert.assertTrue(constraint.getNorm() < 10e-5);
    }

    @Test
    public void convergenceConstraintVertical() {
        Point p10 = new Point(1, 0.0, 0.0);
        Point p20 = new Point(2, 40.0, 40.0);

        Line f10 = new Line(p10, p20);
        f10.setAssociateConstraints(null);

        Constraint constraint = new ConstraintVertical(Constraint.nextId(), p10, p20);

        geometricSolver.solveSystem(solverStat);

        Assert.assertEquals(0, solverStat.iterations);
        Assert.assertTrue(solverStat.convergence);
        Assert.assertTrue(solverStat.delta < 10e-2);
        Assert.assertTrue(solverStat.constraintDelta < 10e-5);
        Assert.assertTrue(constraint.getNorm() < 10e-5);
    }

    @Test
    public void convergenceConstraintLinesParallelism() {
        Point p10 = new Point(1, 0.0, 40.0);
        Point p20 = new Point(2, 10.0, 80.0);
        Point p30 = new Point(5, 40.0, 0.0);
        Point p40 = new Point(6, 90.0, 10.0);

        Line f10 = new Line(p10, p20);
        Line f20 = new Line(p30, p40);
        f10.setAssociateConstraints(null);
        f20.setAssociateConstraints(null);

        Constraint constraint = new ConstraintLinesParallelism(Constraint.nextId(), p10, p20, p30, p40); /// FIXME HESSIAN "unstable"

        geometricSolver.solveSystem(solverStat);

        Assert.assertEquals(4, solverStat.iterations);
        Assert.assertTrue(solverStat.convergence);
        Assert.assertTrue(solverStat.delta < 10e-2);
        Assert.assertTrue(solverStat.constraintDelta < 10e-5);
        Assert.assertTrue(constraint.getNorm() < 10e-5);
    }

    @Test
    public void convergenceConstraintEqualLength() {
        Point p10 = new Point(1, 0.0, 40.0);
        Point p20 = new Point(2, 10.0, 80.0);
        Point p30 = new Point(5, 40.0, 0.0);
        Point p40 = new Point(6, 190.0, 10.0);

        Line f10 = new Line(p10, p20);
        Line f20 = new Line(p30, p40);
        f10.setAssociateConstraints(null);
        f20.setAssociateConstraints(null);

        Constraint constraint = new ConstraintEqualLength(Constraint.nextId(), p10, p20, p30, p40);

        geometricSolver.solveSystem(solverStat);

        Assert.assertEquals(0, solverStat.iterations);
        Assert.assertTrue(solverStat.convergence);
        Assert.assertTrue(solverStat.delta < 10e-2);
        Assert.assertTrue(solverStat.constraintDelta < 10e-5);
        Assert.assertTrue(constraint.getNorm() < 10e-5);
    }


    @Test
    public void convergenceConstraintParametrizedLength() {
        Point p10 = new Point(1, 0.0, 40.0);
        Point p20 = new Point(2, 10.0, 80.0);
        Point p30 = new Point(5, 40.0, 0.0);
        Point p40 = new Point(6, 190.0, 10.0);

        Line f10 = new Line(p10, p20);
        Line f20 = new Line(p30, p40);
        f10.setAssociateConstraints(null);
        f20.setAssociateConstraints(null);
        Parameter param = new Parameter(2.5);
        Constraint constraint = new ConstraintParametrizedLength(Constraint.nextId(), p10, p20, p30, p40, param);

        geometricSolver.solveSystem(solverStat);

        Assert.assertEquals(0, solverStat.iterations);
        Assert.assertTrue(solverStat.convergence);
        Assert.assertTrue(solverStat.delta < 10e-10);
        Assert.assertTrue(solverStat.constraintDelta < 10e-10);
        Assert.assertTrue(constraint.getNorm() < 10e-10);
    }

    @Test
    public void convergenceConstraintDistance2Points() {
        Point p10 = new Point(1, 0.0, 40.0);
        Point p20 = new Point(2, 10.0, 80.0);

        Line f10 = new Line(p10, p20);
        f10.setAssociateConstraints(null);
        Parameter param = new Parameter(200.0);
        Constraint constraint = new ConstraintDistance2Points(Constraint.nextId(), p10, p20, param);

        geometricSolver.solveSystem(solverStat);

        Assert.assertEquals(0, solverStat.iterations);
        Assert.assertTrue(solverStat.convergence);
        Assert.assertTrue(solverStat.delta < 10e-2);
        Assert.assertTrue(solverStat.constraintDelta < 10e-5);
        Assert.assertTrue(constraint.getNorm() < 10e-5);
    }

    @Test
    public void convergenceConstraintAngle2Lines() {
        Point p10 = new Point(1, 0.0, 00.0);
        Point p20 = new Point(2, 10.0, 80.0);
        Point p30 = new Point(5, 10.0, 10.0);
        Point p40 = new Point(6, 90.0, 10.0);

        Line f10 = new Line(p10, p20);
        Line f20 = new Line(p30, p40);
        f10.setAssociateConstraints(null);
        f20.setAssociateConstraints(null);

        Parameter param = new Parameter(30.0);// 30deg
        Constraint constraint = new ConstraintAngle2Lines(Constraint.nextId(), p10, p20, p30, p40, param);

        geometricSolver.solveSystem(solverStat);

        Assert.assertEquals(2, solverStat.iterations);
        Assert.assertTrue(solverStat.convergence);
        Assert.assertTrue(solverStat.delta < 10e-5);
        Assert.assertTrue(solverStat.constraintDelta < 10e-5);
        Assert.assertTrue(constraint.getNorm() < 10e-5);
    }

    @Test
    public void convergenceConstraintAngle2Lines_WithFixedArm() {
        Point p10 = new Point(1, 0.0, 00.0);
        Point p20 = new Point(2, 10.0, 80.0);
        Point p30 = new Point(5, 10.0, 10.0);
        Point p40 = new Point(6, 90.0, 10.0);

        Line f10 = new Line(p10, p20);
        Line f20 = new Line(p30, p40);
        f10.setAssociateConstraints(null);
        f20.setAssociateConstraints(null);

        Parameter param = new Parameter(45.0);// 45deg
        Constraint fixedArm = new ConstraintVertical(Constraint.nextId(), p10, p20);
        Constraint constraint = new ConstraintAngle2Lines(Constraint.nextId(), p10, p20, p30, p40, param);

        geometricSolver.solveSystem(solverStat);

        Assert.assertEquals(3, solverStat.iterations);
        Assert.assertTrue(solverStat.convergence);
        Assert.assertTrue(solverStat.delta < 10e-2);
        Assert.assertTrue(solverStat.constraintDelta < 10e-5);
        Assert.assertTrue(constraint.getNorm() < 10e-5);
    }

    @Test
    public void convergenceConstraintTangency() {
        /*
         * Macierz Hessian'a ?
         */
        Point p10 = new Point(1, 0.0, 00.0);
        Point p20 = new Point(2, 10.0, 80.0);
        Point p30 = new Point(5, 60.0, 60.0);
        Point p40 = new Point(6, 120.0, 120.0);

        Line f10 = new Line(p10, p20);
        Circle f20 = new Circle(p30, p40);
        f10.setAssociateConstraints(null);
        f20.setAssociateConstraints(null);

        Constraint constraint = new ConstraintTangency(Constraint.nextId(), p10, p20, p30, p40);

        geometricSolver.solveSystem(solverStat);

        Assert.assertEquals(19, solverStat.iterations);
        Assert.assertTrue(!solverStat.convergence);
        Assert.assertTrue(solverStat.delta < 10e-2);
        Assert.assertTrue(solverStat.constraintDelta < 10e-2);
        Assert.assertTrue(constraint.getNorm() < 1e-2);
    }

    @Test
    public void convergenceConstraintDistancePointLine() {
        /*
         * Macierz Hessian'a ?
         */
        Point p10 = new Point(1, 0.0, 00.0);
        Point p20 = new Point(2, 0.0, 80.0);
        Point p30 = new Point(5, 60.0, 60.0);

        Parameter param = new Parameter(30.0);// 30 units of distance

        Line f10 = new Line(p10, p20);
        FreePoint f30 = new FreePoint(p30);
        f10.setAssociateConstraints(null);
        f30.setAssociateConstraints(null);

        Constraint constraint = new ConstraintDistancePointLine(Constraint.nextId(), p10, p20, p30, param);

        geometricSolver.solveSystem(solverStat);


        /* no Hessian implementations evaluation into closing equation */

        Assert.assertEquals(19, solverStat.iterations);
//        Assert.assertTrue(!solverStat.convergence);
        Assert.assertTrue(solverStat.delta < 1e-2);
        Assert.assertTrue(solverStat.constraintDelta < 1.0e-2);
        Assert.assertTrue(constraint.getNorm() < 10e-3);
    }

    @Test
    public void testWriteStdOutTensor() {
        Point p10 = new Point(1, 0.0, 00.0);
        Point p20 = new Point(2, 10.0, 80.0);
        Point p30 = new Point(5, 60.0, 60.0);
        Point p40 = new Point(6, 120.0, 120.0);

        Line f10 = new Line(p10, p20);
        Circle f20 = new Circle(p30, p40);
        f10.setAssociateConstraints(null);
        f20.setAssociateConstraints(null);

        Constraint constraint = new ConstraintTangency(Constraint.nextId(), p10, p20, p30, p40);

        out.println(constraint.getJacobian().toString());
        out.println(constraint.getHessian(1.0).toString());
    }

}