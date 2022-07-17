package com.mstruzek.msketch.solver;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.LUDecompositionQuick;
import com.mstruzek.msketch.Constraint;
import com.mstruzek.msketch.GeometricPrimitive;
import com.mstruzek.msketch.ParseToColt;
import com.mstruzek.msketch.Point;
import com.mstruzek.msketch.matrix.BindMatrix;
import com.mstruzek.msketch.matrix.ColtMatrixCreator;
import com.mstruzek.msketch.matrix.MatrixDouble;
import com.mstruzek.msketch.matrix.MatrixDoubleCreator;

import java.time.Clock;

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

        long startTime;                 /// start timing
        long evaluationStart;           /// matrix and vector state evaluation time
        long solverStep;                /// single LU solver round  delta

        final int size;                 /// wektor stanu
        final int coffSize;             /// wspolczynniki Lagrange
        final int dimension;            /// dimension = size + coffSize

        /// Uklad rownan liniowych  [ A * x = b ] powstały z linerazycji ukladu dynamicznego

        final MatrixDouble A;               /// Macierz głowna ukladu rownan liniowych
        final MatrixDouble Fq;              /// Macierz sztywnosci ukladu obiektow zawieszonych na sprezynach.
        final MatrixDouble Wq;              /// d(FI)/dq - Jacobian Wiezow

        /// HESSIAN
        final MatrixDouble Hs;

        // Wektor prawych stron [Fr; Fi]'
        final MatrixDouble b;

        // skladowe to Fr - oddzialywania pomiedzy obiektami, sily w sprezynach
        final MatrixDouble Fr;

        // skladowa to Fiq - wartosci poszczegolnych wiezow
        final MatrixDouble Fi;


        double norm1;               /// wartosci bledow na wiezach
        double prevNorm;            /// norma z wczesniejszej iteracji,
        double errorFluctuation;    /// fluktuacja bledu

        /// Default Matrix Creator for middle matrices !
        MatrixDoubleCreator.setInstance(ColtMatrixCreator.INSTANCE);

        StateReporter reporter = StateReporter.getInstance();

        reporter.writelnf("@#=================== Solver Initialized ===================#@ ");
        reporter.writelnf("");

        startTime = clock.millis();
        solverStat.startTime = startTime;

        if (dbPoint.size() == 0) {
            reporter.writelnf("[warning] - empty model");
            return solverStat;
        }

        if (Constraint.dbConstraint.isEmpty()) {
            reporter.writelnf("[warning] - no constraint configuration applied ");
            return solverStat;
        }

        /// Tworzymy Macierz "A" - dla tego zadania stala w czasie

        size = Point.dbPoint.size() * 2;
        coffSize = Constraint.allLagrangeCoffSize();
        dimension = size + coffSize;

        solverStat.size = size;
        solverStat.coefficientArity = coffSize;
        solverStat.dimension = dimension;

        evaluationStart = clock.millis();

        /// Inicjalizacje bazowych macierzy rzadkich - SparseMatrix

        A = MatrixDouble.matrix2D(dimension, dimension, 0.0);
        Fq = MatrixDouble.matrix2D(size, size, 0.0);
        Wq = MatrixDouble.matrix2D(coffSize, size, 0.0);

        Hs = MatrixDouble.matrix2D(size, size, 0.0);

        /// macierz sztywnosci stala w czasie
        GeometricPrimitive.getAllJacobianForces(Fq);

/// Wektor prawych stron b

        b = MatrixDouble.matrix1D(dimension, 0.0);
    // right-hand side vector ~ b
        Fr = b.viewSpan(0, 0, size, 1);
        Fi = b.viewSpan(size, 0, coffSize, 1);


        BindMatrix dmx = null;

        BindMatrix MTQ = new BindMatrix(dimension, 1);
        MTQ.bindDbPoints();

        solverStat.accEvaluationTime += clock.millis() - evaluationStart;

        norm1 = 0.0;
        prevNorm = 0.0;
        errorFluctuation = 0.0;

        int itr = 0;
        while (itr < MAX_SOLVER_ITERATIONS) {

            solverStat.iterations = itr;

            evaluationStart = clock.millis();

            /// zerujemy macierz A
            A.reset(0.0);

/// Tworzymy Macierz vector b vector `b

            GeometricPrimitive.getAllForce(Fr);                 /// Sily  - F(q)
            Constraint.getFullConstraintValues(Fi);             /// Wiezy  - Fi(q)
            // b.setSubMatrix(0,0, (Fr));
            // b.setSubMatrix(size,0, (Fi));
            b.dot(-1);

/// macierz `A
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
             *   rozwiazjemy zadanie [ A ] * [ dx ] = [ b ]
             */

/// Solver LU Single Iteration Step

            DoubleMatrix2D matrix2DA = ParseToColt.toSparse(A);
            DoubleMatrix1D matrix1Db = ParseToColt.toDenseVector(b);
            solverStep = clock.millis();

            reporter.debug(() -> A.toString(new Integer[0]));
            reporter.debug(() -> b.toString(new Integer[0]));

            {
                LUDecompositionQuick LU = new LUDecompositionQuick();
                LU.decompose(matrix2DA);
                LU.solve(matrix1Db);
            }
            solverStat.accSolverTime += (clock.millis() - solverStep);

/// Bind delta-x into database points
            dmx = ParseToColt.toBindVector(matrix1Db);

            /// uaktualniamy punkty
            MTQ.plusEquals(dmx);
            MTQ.copyToPoints();

            norm1 = Constraint.getFullNorm();

            reporter.writelnf(" [ step :: %d]  duration [ms] = %d  norm = %e ", itr, (clock.millis() - evaluationStart), norm1);

            /// Gdy po 5-6 przejsciach iteracji, normy wiezow kieruja sie w strone minimum energii, to repozycjonowac prowadzace punkty

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
            ///  ######### 1 . przeprowadzimy snapshot punktow w czasie i przy narastajacym bledzie - revert(Guide Points) i kontynuacja !
            ///
            ///  ######### 2. evaluacja bledow wzgledem dominujecej skladowej wiezu NORM !
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

        long solutionDelta = (clock.millis() - startTime);
        reporter.writeln("# solution delta : " + solutionDelta); // print execution time
        reporter.writeln(""); // print execution time

        solverStat.constraintDelta = Constraint.getFullNorm();
        solverStat.convergence = norm1 < CONVERGENCE_LIMIT;
        solverStat.stopTime = clock.millis();

        return solverStat;
    }
}

