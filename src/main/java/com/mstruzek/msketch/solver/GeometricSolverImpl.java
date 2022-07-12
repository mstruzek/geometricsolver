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

    public static final int MAX_SOLVER_ITERATIONS = 15;

    private static final Clock clock = Clock.systemUTC();


    @Override
    public SolverStat solveSystem(StateReporter reporter, SolverStat solverStat) {

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

        MatrixDouble A = MatrixDouble.fill(dimension, dimension, 0.0);
        MatrixDouble Fq = MatrixDouble.fill(size, size, 0.0);               // CONST
        MatrixDouble Wq = MatrixDouble.fill(coffSize, size, 0.0);

        /// HESSIAN
        MatrixDouble Hs = MatrixDouble.fill(size, size, 0.0);


        // right-hand side vector ~ b
        MatrixDouble Fr = MatrixDouble.fill(size, 1, 0.0);
        MatrixDouble Fi = MatrixDouble.fill(coffSize, 1, 0.0);

        /// CONST
        GeometricPrimitive.getAllJacobianForces(Fq);

        // Tworzymy wektor prawych stron b
        MatrixDouble b = null;
        BindMatrix dmx = null;

        BindMatrix Bmt = new BindMatrix(dimension, 1);
        Bmt.bind(dbPoint);

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

            b = MatrixDouble.mergeByColumn((Fr), (Fi));
            b.dot(-1);

            /// JACOBIAN
            Constraint.getFullJacobian(Wq);                     /// Jq = d(Fi)/dq

            Hs.reset(0.0);

            Constraint.getFullHessian(Hs, Bmt);

            A.setSubMatrix(0, 0, Fq);                   /// procedure SET
            A.addSubMatrix(0, 0, Hs);                   /// procedure ADD

            A.setSubMatrix(size, 0, Wq);
            A.setSubMatrix(0, size, Wq.transpose());

            solverStat.accEvaluationTime += (clock.millis() - evaluationStart);

            /*
             *  LU Decomposition  -- Colt Linera Equatio Solver
             *
             */
            /// rozwiazjemy zadanie A*dx=b
            /// Zaproponowac Wspolny interfejs Macierzowy

/// [ TIMED ]
            long solverStep = clock.millis(); // start timing

            DoubleMatrix2D matrix2DA = ParseToColt.toSparse(A);
            DoubleMatrix1D matrix1Db = ParseToColt.toDenseVector(b);

            {
                LUDecompositionQuick LU = new LUDecompositionQuick();
                LU.decompose(matrix2DA);
                LU.solve(matrix1Db);
            }
            solverStat.accSolverTime += (clock.millis() - solverStep);

/// [ TIMED ]

            dmx = ParseToColt.toBindVector(matrix1Db);

            /// uaktualniamy punkty
            Bmt.plusEquals(dmx);
            Bmt.copyToPoints();

            /// AFTER -- copyToPoints  x2
            Constraint.getFullConstraintValues(Fi);

            norm1 = Fi.norm1();

            reporter.writelnf(" [ step :: %d]  duration [ms] = %d  norm = %e ", itr, (clock.millis() - solverStep), norm1);

            //stary warunek wyjscia
            if (norm1 < 10e-5) {
                solverStat.delta = norm1;
                reporter.writelnf("fast convergence - norm [ %e ]  , constraint error = %e", norm1, Constraint.getFullNorm());
                reporter.writeln("");
                break;
            }

            /// liczymy zmiane bledu
            errorFluctuation = norm1 - prevNorm;
            prevNorm = norm1;

            solverStat.delta = norm1;

            if (itr > 1 && errorFluctuation/prevNorm > 0.70) {
                reporter.writeln("CHANGES - STOP ITERATION *******");
                reporter.writeln(" errorFluctuation          :" + errorFluctuation);
                reporter.writeln(" relative error            :" + (errorFluctuation/norm1));
                solverStat.constraintDelta = Constraint.getFullNorm();
                solverStat.converged = false;
                solverStat.stopTime = clock.millis();
                return solverStat;
            }

            itr++;
        }

        long solutionDelta = (clock.millis() - start);
        reporter.writeln("# solution error : " + solutionDelta); // print execution time
        reporter.writeln(""); // print execution time

        solverStat.constraintDelta = Constraint.getFullNorm();
        solverStat.converged = true;
        solverStat.stopTime = clock.millis();

        return solverStat;
    }
}
