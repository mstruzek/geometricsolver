package com.mstruzek.msketch;

import com.mstruzek.msketch.matrix.MatrixDouble;
import com.mstruzek.msketch.solver.GeometricSolver;
import com.mstruzek.msketch.solver.GeometricSolverImpl;
import com.mstruzek.msketch.solver.SolverStat;
import com.mstruzek.msketch.solver.StateReporter;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import static com.mstruzek.msketch.ModelRegistry.nextParameterId;
import static java.lang.System.out;

public class ConstraintConvergenceTest {

    SolverStat solverStat;
    StateReporter reporter;
    GeometricSolver geometricSolver;

    @Before
    public void beforeTest() {
        reporter = StateReporter.getInstance();
        solverStat = new SolverStat();
        geometricSolver = new GeometricSolverImpl();
        geometricSolver.setup();
    }

    @After
    public void afterTest() {
        reporter.reportSolverStatistics(solverStat);

        ModelRegistry.removeObjectsFromModel();
    }

    private static <T extends GeometricObject> T registerGeometric(T geometricObject) {
        Model.add(geometricObject);
        return geometricObject;
    }

    private static <C extends Constraint> C registerConstraint(C constraint) {
        Model.add(constraint);
        return constraint;
    }

    private static Parameter registerParameter(Parameter parameter) {
        Model.add(parameter);
        return parameter;
    }

    @Test
    public void convergenceConstraintFixPoint() {
        Point p10 = new Point(1, 0.0, 40.0);
        FreePoint f10 = new FreePoint(p10);
        f10.setAssociateConstraints(null);
        registerGeometric(f10);

        Constraint constraint = registerConstraint(new ConstraintFixPoint(ModelRegistry.nextConstraintId(), p10));
        p10.setLocation(100.0, 100.0);

        geometricSolver.solveSystem(solverStat);

        Assert.assertEquals(0, solverStat.iterations);
        Assert.assertTrue(solverStat.convergence);
        Assert.assertTrue(solverStat.error < 10e-10);
        Assert.assertTrue(solverStat.constraintDelta < 10e-10);
        Assert.assertTrue(constraint.getNorm() < 10e-10);
    }

    @Test
    public void convergenceConstraintParametrizedXFix() {
        Point p10 = new Point(1, 40.0, 40.0);
        FreePoint f10 = new FreePoint(p10);
        f10.setAssociateConstraints(null);
        registerGeometric(f10);
        Parameter parameter = registerParameter(new Parameter(nextParameterId(), 120.0)); // fixed X coordinate
        Constraint constraint = registerConstraint(new ConstraintParametrizedXFix(ModelRegistry.nextConstraintId(), p10, parameter));

        geometricSolver.solveSystem(solverStat);

        Assert.assertEquals(0, solverStat.iterations);
        Assert.assertTrue(solverStat.convergence);
        Assert.assertTrue(solverStat.error < 10e-10);
        Assert.assertTrue(solverStat.constraintDelta < 10e-10);
        Assert.assertTrue(constraint.getNorm() < 10e-10);
        Assert.assertEquals(p10.getX(), parameter.getValue(), 10e-10);
    }

    @Test
    public void convergenceConstraintParametrizedYFix() {
        Point p10 = new Point(1, 40.0, 40.0);
        FreePoint f10 = new FreePoint(p10);
        f10.setAssociateConstraints(null);
        registerGeometric(f10);
        Parameter parameter = registerParameter(new Parameter(nextParameterId(), 120.0)); // fixed Y coordinate
        Constraint constraint = registerConstraint(new ConstraintParametrizedYFix(ModelRegistry.nextConstraintId(), p10, parameter));

        geometricSolver.solveSystem(solverStat);

        Assert.assertEquals(0, solverStat.iterations);
        Assert.assertTrue(solverStat.convergence);
        Assert.assertTrue(solverStat.error < 10e-10);
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
        registerGeometric(f10);
        registerGeometric(f20);
        Constraint constraint = registerConstraint(new ConstraintConnect2Points(ModelRegistry.nextConstraintId(), p10, p20));

        geometricSolver.solveSystem(solverStat);

        Assert.assertEquals(0, solverStat.iterations);
        Assert.assertTrue(solverStat.convergence);
        Assert.assertTrue(solverStat.error < 10e-10);
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
        registerGeometric(f10);
        registerGeometric(f20);

        Constraint constraint = registerConstraint(new ConstraintLinesPerpendicular(ModelRegistry.nextConstraintId(), p10, p20, p30, p40));

        geometricSolver.solveSystem(solverStat);

        Assert.assertEquals(2, solverStat.iterations);
        Assert.assertTrue(solverStat.convergence);
        Assert.assertTrue(solverStat.error < 10e-2);
        Assert.assertTrue(solverStat.constraintDelta < 10e-5);
        Assert.assertTrue(constraint.getNorm() < 10e-5);
    }

    @Test
    public void convergenceConstraintHorizontal() {
        Point p10 = new Point(1, 0.0, 0.0);
        Point p20 = new Point(2, 40.0, 40.0);

        Line f10 = new Line(p10, p20);
        f10.setAssociateConstraints(null);
        registerGeometric(f10);

        Constraint constraint = registerConstraint(new ConstraintHorizontal(ModelRegistry.nextConstraintId(), p10, p20));

        geometricSolver.solveSystem(solverStat);

        Assert.assertEquals(0, solverStat.iterations);
        Assert.assertTrue(solverStat.convergence);
        Assert.assertTrue(solverStat.error < 10e-2);
        Assert.assertTrue(solverStat.constraintDelta < 10e-5);
        Assert.assertTrue(constraint.getNorm() < 10e-5);
    }

    @Test
    public void convergenceConstraintVertical() {
        Point p10 = new Point(1, 0.0, 0.0);
        Point p20 = new Point(2, 40.0, 40.0);

        Line f10 = new Line(p10, p20);
        f10.setAssociateConstraints(null);
        registerGeometric(f10);

        Constraint constraint = registerConstraint(new ConstraintVertical(ModelRegistry.nextConstraintId(), p10, p20));

        geometricSolver.solveSystem(solverStat);

        Assert.assertEquals(0, solverStat.iterations);
        Assert.assertTrue(solverStat.convergence);
        Assert.assertTrue(solverStat.error < 10e-2);
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
        registerGeometric(f10);
        registerGeometric(f20);

        Constraint constraint = registerConstraint(new ConstraintLinesParallelism(ModelRegistry.nextConstraintId(), p10, p20, p30, p40)); /// FIXME HESSIAN "unstable"

        geometricSolver.solveSystem(solverStat);

        Assert.assertEquals(4, solverStat.iterations);
        Assert.assertTrue(solverStat.convergence);
        Assert.assertTrue(solverStat.error < 10e-2);
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
        registerGeometric(f10);
        registerGeometric(f20);

        Constraint constraint = registerConstraint(new ConstraintEqualLength(ModelRegistry.nextConstraintId(), p10, p20, p30, p40));

        geometricSolver.solveSystem(solverStat);

        Assert.assertEquals(0, solverStat.iterations);
        Assert.assertTrue(solverStat.convergence);
        Assert.assertTrue(solverStat.error < 10e-2);
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
        registerGeometric(f10);
        registerGeometric(f20);

        Parameter param = registerParameter(new Parameter(nextParameterId(), 2.5));
        Constraint constraint = registerConstraint(new ConstraintParametrizedLength(ModelRegistry.nextConstraintId(), p10, p20, p30, p40, param));

        geometricSolver.solveSystem(solverStat);

        Assert.assertEquals(0, solverStat.iterations);
        Assert.assertTrue(solverStat.convergence);
        Assert.assertTrue(solverStat.error < 10e-10);
        Assert.assertTrue(solverStat.constraintDelta < 10e-10);
        Assert.assertTrue(constraint.getNorm() < 10e-10);
    }

    @Test
    public void convergenceConstraintDistance2Points() {
        Point p10 = new Point(1, 0.0, 40.0);
        Point p20 = new Point(2, 10.0, 80.0);

        Line f10 = new Line(p10, p20);
        f10.setAssociateConstraints(null);
        registerGeometric(f10);

        Parameter param = registerParameter(new Parameter(nextParameterId(), 200.0));
        Constraint constraint = registerConstraint(new ConstraintDistance2Points(ModelRegistry.nextConstraintId(), p10, p20, param));

        geometricSolver.solveSystem(solverStat);

        Assert.assertEquals(0, solverStat.iterations);
        Assert.assertTrue(solverStat.convergence);
        Assert.assertTrue(solverStat.error < 10e-2);
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
        registerGeometric(f10);
        registerGeometric(f20);

        Parameter param = registerParameter(new Parameter(nextParameterId(), 30.0));// 30deg
        Constraint constraint = registerConstraint(new ConstraintAngle2Lines(ModelRegistry.nextConstraintId(), p10, p20, p30, p40, param));

        geometricSolver.solveSystem(solverStat);

        Assert.assertEquals(2, solverStat.iterations);
        Assert.assertTrue(solverStat.convergence);
        Assert.assertTrue(solverStat.error < 10e-5);
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
        registerGeometric(f10);
        registerGeometric(f20);

        Parameter param = registerParameter(new Parameter(nextParameterId(), 45.0));// 45deg
        Constraint fixedArm = registerConstraint(new ConstraintVertical(ModelRegistry.nextConstraintId(), p10, p20));
        Constraint constraint = registerConstraint(new ConstraintAngle2Lines(ModelRegistry.nextConstraintId(), p10, p20, p30, p40, param));

        geometricSolver.solveSystem(solverStat);

        Assert.assertEquals(3, solverStat.iterations);
        Assert.assertTrue(solverStat.convergence);
        Assert.assertTrue(solverStat.error < 10e-2);
        Assert.assertTrue(solverStat.constraintDelta < 10e-5);
        Assert.assertTrue(constraint.getNorm() < 10e-5);
    }

    @Test
    public void convergenceConstraintTangency() {
        Point p10 = new Point(1, 0.0, 00.0);
        Point p20 = new Point(2, 10.0, 80.0);
        Point p30 = new Point(5, 60.0, 60.0);
        Point p40 = new Point(6, 120.0, 120.0);

        Line f10 = new Line(p10, p20);
        Circle f20 = new Circle(p30, p40);
        f10.setAssociateConstraints(null);
        f20.setAssociateConstraints(null);
        registerGeometric(f10);
        registerGeometric(f20);

        Constraint constraint = registerConstraint(new ConstraintTangency(ModelRegistry.nextConstraintId(), p10, p20, p30, p40));        /// Macierz Hessian'a ?!!

        geometricSolver.solveSystem(solverStat);

        Assert.assertEquals(20, solverStat.iterations);
        Assert.assertTrue(!solverStat.convergence);
        Assert.assertTrue(solverStat.error < 10e-2);
        Assert.assertTrue(solverStat.constraintDelta < 10e-2);
        Assert.assertTrue(constraint.getNorm() < 1e-2);
    }

    @Test
    public void convergenceConstraintCircleTangency() {
        Point p10 = new Point(1, 0.0, 00.0);
        Point p20 = new Point(2, 10.0, 80.0);
        Point p30 = new Point(5, 60.0, 60.0);
        Point p40 = new Point(6, 120.0, 120.0);

        Circle f10 = new Circle(p10, p20);
        Circle f20 = new Circle(p30, p40);
        f10.setAssociateConstraints(null);
        f20.setAssociateConstraints(null);
        registerGeometric(f10);
        registerGeometric(f20);

        Constraint constraint = registerConstraint(new ConstraintCircleTangency(ModelRegistry.nextConstraintId(), p10, p20, p30, p40));

        geometricSolver.solveSystem(solverStat);

        Assert.assertEquals(0, solverStat.iterations);
        Assert.assertTrue(solverStat.convergence);
        Assert.assertTrue(solverStat.error < 10e-4);
        Assert.assertTrue(solverStat.constraintDelta < 10e-4);
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

        Parameter param = registerParameter(new Parameter(nextParameterId(), 30.0));// 30 units of distance

        Line f10 = new Line(p10, p20);
        FreePoint f30 = new FreePoint(p30);
        f10.setAssociateConstraints(null);
        f30.setAssociateConstraints(null);
        registerGeometric(f10);
        registerGeometric(f30);

        Constraint constraint = registerConstraint(new ConstraintDistancePointLine(ModelRegistry.nextConstraintId(), p10, p20, p30, param));

        geometricSolver.solveSystem(solverStat);


        /* no Hessian implementations evaluation into closing equation */

        Assert.assertEquals(19, solverStat.iterations);
//        Assert.assertTrue(!solverStat.convergence);
        Assert.assertTrue(solverStat.error < 1e-2);
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
        registerGeometric(f10);
        registerGeometric(f20);

        Constraint constraint = registerConstraint(new ConstraintTangency(ModelRegistry.nextConstraintId(), p10, p20, p30, p40));

        geometricSolver.solveSystem(solverStat);

        MatrixDouble matrixDouble = MatrixDouble.matrix2D(1, ModelRegistry.dbPoint.size() * 2, 0.0);
        constraint.getJacobian(matrixDouble);

        out.println(matrixDouble.toString());

        MatrixDouble hessian = constraint.getHessian(1.0);
        if(hessian != null){
            out.println(MatrixDouble.writeToString(hessian));
        }
    }
}