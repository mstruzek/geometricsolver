package com.mstruzek.msketch.solver;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.LUDecompositionQuick;
import com.mstruzek.msketch.Constraint;
import com.mstruzek.msketch.GeometricPrimitive;
import com.mstruzek.msketch.ParseToColt;
import com.mstruzek.msketch.matrix.BindMatrix;
import com.mstruzek.msketch.matrix.MatrixDouble;

import java.time.Clock;

import static com.mstruzek.msketch.Constraint.allLagrangeCoffSize;
import static com.mstruzek.msketch.Point.dbPoint;

public class GeometricSolverImpl implements GeometricSolver {

    // 200x200 macierz A = > 50 okregow/linii  =>   200*200*8 = 320kB

    public static final int MAX_SOLVER_ITERATIONS = 20;

    private static final Clock clock = Clock.systemUTC();

    /**
     * convergence limit
     */
    public static final double CONVERGENCE_LIMIT = 10e-5;

    @Override
    public SolverStat solveSystem(SolverStat solverStat) {

        StateReporter reporter = StateReporter.getInstance();

        reporter.writelnf("##################### Solver Initialized ##################### ");
        reporter.writelnf("");

        long start = clock.millis();                  /// start timing

        solverStat.startTime = start;

        if (dbPoint.size() == 0) {
            return solverStat;
        }

        /// Tworzymy Macierz "A" - dla tego zadania stala w czasie

        /// TODO  @IniticjalizajaMacierzy

        final int size = dbPoint.size() * 2;
        final int coffSize = allLagrangeCoffSize();
        final int dimension = size + coffSize;

        solverStat.size = size;
        solverStat.coefficientArity = coffSize;
        solverStat.dimension = dimension;

        long evaluationStart = clock.millis();

        MatrixDouble A = MatrixDouble.matrix2D(dimension, dimension, 0.0);
        MatrixDouble Fq = MatrixDouble.matrix2D(size, size, 0.0);               // CONST
        MatrixDouble Wq = MatrixDouble.matrix2D(coffSize, size, 0.0);

        /// HESSIAN
        MatrixDouble Hs = MatrixDouble.matrix2D(size, size, 0.0);

        // right-hand side vector ~ b
        MatrixDouble Fr = MatrixDouble.matrix1Dtr(size, 0.0);
        MatrixDouble Fi = MatrixDouble.matrix1Dtr(coffSize, 0.0);

        /// CONST
        GeometricPrimitive.getAllJacobianForces(Fq);

        // Tworzymy wektor prawych stron b
        MatrixDouble b = MatrixDouble.matrix1Dtr(dimension, 0.0);
        BindMatrix dmx = null;

        BindMatrix MTQ = new BindMatrix(dimension, 1);
        MTQ.bindDbPoints();

        solverStat.accEvaluationTime += clock.millis() - evaluationStart;

        double norm1 = 0.0;
        double prevNorm = 0.0;
        double errorFluctuation = 0.0;

        int itr = 0;
        while (itr < MAX_SOLVER_ITERATIONS) {

            solverStat.iterations = itr;

            evaluationStart = clock.millis();
            /// zerujemy macierz A
            A.reset(0.0);

            /// Tworzymy Macierz vector b
            GeometricPrimitive.getAllForce(Fr);                 /// Sily  - F(q)
            Constraint.getFullConstraintValues(Fi);             /// Wiezy  - Fi(q)

            b.setSubMatrix(0,0, (Fr));
            b.setSubMatrix(size,0, (Fi));
            b.dot(-1);

            /// JACOBIAN
            Constraint.getFullJacobian(Wq);                     /// Jq = d(Fi)/dq

            Hs.reset(0.0);

            Constraint.getFullHessian(Hs, MTQ, size);

            A.setSubMatrix(0, 0, Fq);                   /// procedure SET
            A.addSubMatrix(0, 0, Hs);                   /// procedure ADD

            A.setSubMatrix(size, 0, Wq);
            A.setSubMatrix(0, size, Wq.transpose());

            solverStat.accEvaluationTime += (clock.millis() - evaluationStart);

            /*
             *  LU Decomposition  -- Colt Linera Equatio Solver
             *
             */
            /// rozwiazjemy zadanie [ A ] * [ dx ] = [ b ]
            /// Zaproponowac Wspolny interfejs Macierzowy

/// [ TIMED ]
            long solverStep = clock.millis(); // start timing

            DoubleMatrix2D matrix2DA = ParseToColt.toSparse(A);
            DoubleMatrix1D matrix1Db = ParseToColt.toDenseVector(b);
            reporter.debug(A.toString(new Integer[0]));
            reporter.debug(b.toString(new Integer[0]));

            {
                LUDecompositionQuick LU = new LUDecompositionQuick();
                LU.decompose(matrix2DA);
                LU.solve(matrix1Db);
//                System.out.println(LU.toString());
            }
            solverStat.accSolverTime += (clock.millis() - solverStep);

/// [ TIMED ]
            dmx = ParseToColt.toBindVector(matrix1Db);

            /// uaktualniamy punkty
            MTQ.plusEquals(dmx);
            MTQ.copyToPoints();

            /// AFTER -- copyToPoints  x2
//            Constraint.getFullConstraintValues(Fi);
//
//            norm1 = Fi.norm1();

            norm1 = Constraint.getFullNorm();

            reporter.writelnf(" [ step :: %d]  duration [ms] = %d  norm = %e ", itr, (clock.millis() - solverStep), norm1);

            /// Gdy po 5-6 przejsciach iteracji, normy wiezow kieruja sie w strone minimum energii, to repozycjonowac prowadzacych punktow

            if (norm1 < CONVERGENCE_LIMIT) {
                solverStat.delta = norm1;
                reporter.writelnf("fast convergence - norm [ %e ]  ", norm1);
                reporter.writelnf("constraint error = %e", Constraint.getFullNorm());
                reporter.writeln("");
                break;
            }

            /// liczymy zmiane bledu
            errorFluctuation = norm1 - prevNorm;
            prevNorm = norm1;

            solverStat.delta = norm1;

            ///
            ///
            ///  ######### 1 . przeprowadzimy snapshot punktow w czasie i przy narastajacym bledzie - revert(Guide Points) i kontynuacja !
            ///
            ///  ######### 2. evaluacja bledow wzgledem dominujecej skladowej wiezu NORM !
            ///
            ///

            if (itr > 1 && errorFluctuation / prevNorm > 0.70) {
                reporter.writeln("CHANGES - STOP ITERATION *******");
                reporter.writeln(" errorFluctuation          :" + errorFluctuation);
                reporter.writeln(" relative error            :" + (errorFluctuation / norm1));
                solverStat.constraintDelta = Constraint.getFullNorm();
                solverStat.convergence = false;
                solverStat.stopTime = clock.millis();
                return solverStat;
            }

            itr++;
        }

        long solutionDelta = (clock.millis() - start);
        reporter.writeln("# solution error : " + solutionDelta); // print execution time
        reporter.writeln(""); // print execution time

        solverStat.constraintDelta = Constraint.getFullNorm();
        solverStat.convergence = norm1 < CONVERGENCE_LIMIT;
        solverStat.stopTime = clock.millis();

        return solverStat;
    }
}

