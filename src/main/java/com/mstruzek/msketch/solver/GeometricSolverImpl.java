package com.mstruzek.msketch.solver;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.LUDecompositionQuick;
import com.mstruzek.msketch.*;
import com.mstruzek.msketch.matrix.TensorDouble;
import com.mstruzek.msketch.matrix.PointUtility;

import java.time.Instant;
import java.util.function.LongSupplier;

import static com.mstruzek.msketch.ModelRegistry.dbPoint;

public class GeometricSolverImpl implements GeometricSolver {

    // 200x200 macierz A = > 50 okregow/linii  =>   200*200*8 = 320kB

    public static final int MAX_SOLVER_ITERATIONS = 20;

    /*** convergence limit */
    public static final double CONVERGENCE_LIMIT = 10e-5;

    private StateReporter reporter;

    private StopWatch solverWatch;          /// Solver start/stop                            [ns]
    private StopWatch accEvoWatch;          /// Accumulated Evaluation Time - for each round [ns]
    private StopWatch accLUWatch;             /// Accumulated LU Solver Time - for each round  [ns]

    public static class StopWatch {
        private static final LongSupplier nanoClock = System::nanoTime;
        long startTick;
        long stopTick;
        long accTime;

        private StopWatch() {
        }

        public void startTick() {
            this.startTick = nanoClock.getAsLong();
        }

        public void stopTick() {
            this.stopTick = nanoClock.getAsLong();
            this.accTime += (stopTick - startTick);
        }

        public long delta() {
            return stopTick - startTick;
        }
    }


    @Override
    public void setup() {

        StateReporter.DebugEnabled = true;

        reporter = StateReporter.getInstance();

        solverWatch = new StopWatch();
        accEvoWatch = new StopWatch();
        accLUWatch = new StopWatch();

//        Default Matrix Creator for middle matrices !
//        MatrixDoubleCreator.setInstance(ColtMatrixCreator.INSTANCE);  // [ #[ #[ HEAVY ]# ]# ]

    }


    @Override
    public SolverStat solveSystem(SolverStat solverStat) {

        final int size;                 /// wektor stanu
        final int coffSize;             /// wspolczynniki Lagrange
        final int dimension;            /// dimension = size + coffSize

        /// Uklad rownan liniowych  [ A * x = b ] powstały z linerazycji ukladu dynamicznego

        final TensorDouble A;               /// Macierz głowna ukladu rownan liniowych
        final TensorDouble Fq;              /// Macierz sztywnosci ukladu obiektow zawieszonych na sprezynach.
        final TensorDouble Wq;              /// d(FI)/dq - Jacobian Wiezow

        /// HESSIAN
        final TensorDouble Hs;

        // Wektor prawych stron [Fr; Fi]'
        final TensorDouble b;

        // skladowe to Fr - oddzialywania pomiedzy obiektami, sily w sprezynach
        final TensorDouble Fr;

        // skladowa to Fiq - wartosci poszczegolnych wiezow
        final TensorDouble Fi;

        double norm1;               /// wartosci bledow na wiezach
        double prevNorm;            /// norma z wczesniejszej iteracji,
        double errorFluctuation;    /// fluktuacja bledu


        solverStat.startTime  = Instant.now().toEpochMilli();
        reporter.writeln("#=================== Solver Initialized ===================# ");
        reporter.writeln("");

        solverWatch.startTick();


        if (dbPoint.size() == 0) {
            reporter.writeln("[warning] - empty model");
            return solverStat;
        }

        if (ModelRegistry.dbConstraint.isEmpty()) {
            reporter.writeln("[warning] - no constraint configuration applied ");
            return solverStat;
        }

        /// Tworzymy Macierz "A" - dla tego zadania stala w czasie

        size = ModelRegistry.dbPoint.size() * 2;
        coffSize = Constraint.allLagrangeCoffSize();
        dimension = size + coffSize;

        solverStat.size = size;
        solverStat.coefficientArity = coffSize;
        solverStat.dimension = dimension;

        accEvoWatch.startTick();

        /// Inicjalizacje bazowych macierzy rzadkich - SparseMatrix

        /// --->
        // instead loop over vector space provide static access to point location in VS.
        // - reference locations for Jacobian and Hessian evaluation !
        PointLocation.setup();

        A = TensorDouble.matrix2D(dimension, dimension, 0.0);
        Fq = TensorDouble.matrix2D(size, size, 0.0);

        Wq = TensorDouble.matrix2D(coffSize, size, 0.0);
        Hs = TensorDouble.matrix2D(size, size, 0.0);

        /// macierz sztywnosci stala w czasie
        GeometricObject.evaluateStiffnessMatrix(Fq);

/// Wektor prawych stron b

        b = TensorDouble.matrix1D(dimension, 0.0);
        // right-hand side vector ~ b
        Fr = b.viewSpan(0, 0, size, 1);
        Fi = b.viewSpan(size, 0, coffSize, 1);

        TensorDouble dmx = null;

        /// State Vector - zmienne stanu
        TensorDouble SV = TensorDouble.matrix1D(dimension, 0.0);
        PointUtility.copyIntoStateVector(SV);
        PointUtility.setupLagrangeMultipliers(SV);

        accEvoWatch.stopTick();

        norm1 = 0.0;
        prevNorm = 0.0;
        errorFluctuation = 0.0;

        int itr = 0;
        while (itr < MAX_SOLVER_ITERATIONS) {

            accEvoWatch.startTick();

            /// zerujemy macierz A
            A.reset(0.0);

/// Tworzymy Macierz vector b vector `b

            GeometricObject.evaluateForceVector(Fr);                 /// Sily  - F(q)
            Constraint.evaluateConstraintVector(Fi);             /// Wiezy  - Fi(q)
            // b.setSubMatrix(0,0, (Fr));
            // b.setSubMatrix(size,0, (Fi));
            b.mulitply(-1);

/// macierz `A


            /// JACOBIAN
            Constraint.getFullJacobian(Wq);                     /// Jq = d(Fi)/dq

            if (StateReporter.isDebugEnabled()) {
//                reporter.writeln(MatrixDouble.writeToString(Wq));
            }

            Hs.reset(0.0);

            Constraint.getFullHessian(Hs, SV, size);

            A.setSubMatrix(0, 0, Fq);                   /// procedure SET
            A.plusSubMatrix(0, 0, Hs);                   /// procedure ADD

            A.setSubMatrix(size, 0, Wq);
            A.setSubMatrix(0, size, Wq.transpose());

            /*
             *  LU Decomposition  -- Colt Linear Equation Solver
             *
             *   rozwiazujemy zadanie [ A ] * [ dx ] = [ b ]
             */
/// Solver LU Single Iteration Step

            if (StateReporter.isDebugEnabled()) {
                reporter.writeln(TensorDouble.writeToString(A));
            }

            DoubleMatrix2D matrix2DA = MatrixDoubleUtility.toSparse(A);
            DoubleMatrix1D matrix1Db = MatrixDoubleUtility.toDenseVector(b);

            accEvoWatch.stopTick();

            accLUWatch.startTick();

            if (StateReporter.isDebugEnabled()) {
//                reporter.writeln(A.toString(new Integer[0]));
                //reporter.writeln(b.toString(new Integer[0]));
            }

/// LU Solver
            LUDecompositionQuick LU = new LUDecompositionQuick();
            LU.decompose(matrix2DA);
            LU.solve(matrix1Db);

            accLUWatch.stopTick();

/// Bind delta-x into database points

            dmx = TensorDouble.matrixDoubleFrom(matrix1Db);

            /// uaktualniamy punkty [ SV ] = [ SV ] + [ delta ]
            SV.plus(dmx);
            PointUtility.copyFromStateVector(SV);


            // Lagrange Coefficients
            TensorDouble LC = SV.viewSpan(size, 0, coffSize, 1);
            if (StateReporter.isDebugEnabled()) {
                reporter.writeln(TensorDouble.writeToString(LC));
            }


            norm1 = Constraint.getFullNorm();

            reporter.writelnf(" [ step :: %02d]  duration [ns] = %,12d  norm = %e ", itr, (accLUWatch.stopTick - accEvoWatch.startTick), norm1);

            /// Gdy po 5-6 przejsciach iteracji, normy wiezow kieruja sie w strone minimum energii, to repozycjonowac prowadzace punkty

            if (norm1 < CONVERGENCE_LIMIT) {
                solverStat.error = norm1;
                reporter.writelnf("fast convergence - norm [ %e ]  ", norm1);
                reporter.writelnf("constraint error = %e", Constraint.getFullNorm());
                reporter.writeln("");
                break;
            }

            /// liczymy zmiane bledu
            errorFluctuation = norm1 - prevNorm;
            prevNorm = norm1;
            solverStat.error = norm1;

            ///
            ///  ######### 1 . przeprowadzimy snapshot punktow w czasie i przy narastajacym bledzie - revert(Guide Points) i kontynuacja !
            ///
            ///  ######### 2. evaluacja bledow wzgledem dominujecej skladowej wiezu NORM !
            ///
            if (itr > 1 && errorFluctuation / prevNorm > 0.70) {
                reporter.writeln("CHANGES - STOP ITERATION *******");
                reporter.writeln(" errorFluctuation          :" + errorFluctuation);
                reporter.writeln(" relative error            :" + (errorFluctuation / norm1));
                solverWatch.stopTick();
                solverStat.constraintDelta = Constraint.getFullNorm();
                solverStat.convergence = false;
                solverStat.stopTime = Instant.now().toEpochMilli();
                solverStat.iterations = itr;
                solverStat.accSolverTime = accLUWatch.accTime;
                solverStat.accEvaluationTime = accEvoWatch.accTime;
                solverStat.timeDelta = solverWatch.stopTick - solverWatch.startTick;
                return solverStat;
            }
            itr++;
        }

        solverWatch.stopTick();
        long solutionDelta = solverWatch.delta();

        reporter.writeln("# solution delta : " + solutionDelta); // print execution time
        reporter.writeln(""); // print execution time

        solverStat.constraintDelta = Constraint.getFullNorm();
        solverStat.convergence = norm1 < CONVERGENCE_LIMIT;
        solverStat.stopTime = Instant.now().toEpochMilli();
        solverStat.iterations = itr;
        solverStat.accSolverTime = accLUWatch.accTime;
        solverStat.accEvaluationTime = accEvoWatch.accTime;
        solverStat.timeDelta = solverWatch.stopTick - solverWatch.startTick;
        return solverStat;
    }

    public static void main(String[] args) {

    }
}

