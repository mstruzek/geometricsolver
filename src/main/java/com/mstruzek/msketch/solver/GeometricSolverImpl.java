package com.mstruzek.msketch.solver;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.LUDecompositionQuick;
import com.mstruzek.msketch.*;
import com.mstruzek.msketch.matrix.*;

import java.time.Instant;
import java.util.concurrent.*;

import static com.mstruzek.msketch.ModelRegistry.dbPoint;

public class GeometricSolverImpl implements GeometricSolver {

    // 200x200 macierz A = > 50 okregow/linii  =>   200*200*8 = 320kB

    public static final int MAX_SOLVER_ITERATIONS = 20;

    /*** convergence limit */
    public static final double CONVERGENCE_LIMIT = 10e-5;

    private StateReporter reporter;

    private StopWatch solverWatch = new StopWatch();          /// Solver start/stop                            [ns]
    private StopWatch accEvoWatch = new StopWatch();          /// Accumulated Evaluation Time - for each round [ns]
    private StopWatch accLUWatch = new StopWatch();             /// Accumulated LU Solver Time - for each round  [ns]


    private ExecutorService executorService;

    @Override
    public GeometricSolverType solverType() {
        return GeometricSolverType.CPU_SOLVER;
    }

    @Override
    public void initializeDriver() {
        ///
        if (executorService == null) {
//            executorService = Executors.newSingleThreadExecutor();
            executorService = Executors.newWorkStealingPool();
        }
    }

    @Override
    public void setup() {
        StateReporter.DebugEnabled = true;

        reporter = StateReporter.getInstance();

        solverWatch.reset();
        accEvoWatch.reset();
        accLUWatch.reset();

        //        Default Matrix Creator for middle matrices !
        MatrixDoubleCreator.setInstance(ColtMatrixCreator.INSTANCE);  // [ #[ #[ HEAVY ]# ]# ]
        /// tolerance for zero elements !
    }


    @Override
    public SolverStat solveSystem() {

        final int size;                 /// wektor stanu
        final int coffSize;             /// wspolczynniki Lagrange
        final int dimension;            /// dimension = size + coffSize

        /// Uklad rownan liniowych  [ A * x = b ] powstały z linerazycji ukladu dynamicznego

        final TensorDouble A;               /// Macierz głowna ukladu rownan liniowych
        final TensorDouble Fq;              /// Macierz sztywnosci ukladu obiektow zawieszonych na sprezynach.
        final TensorDouble Wq;              /// d(FI)/dq - Jacobian Wiezow
        final TensorDouble Mcf;              /// Macierz dopelnien zerami - coefficients size

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


        SolverStat solverStat = new SolverStat();

        solverStat.startTime = Instant.now().toEpochMilli();
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

        /*
         * Executor - Scheduler - A scheduler , B scheduler - ForkJoinPoll
         */
        DoubleMatrix2D.SET_QUICK_ZERO_TOLERANCE = true;     /// setQuick - tolerance
        final boolean NON_ZERO_CAPTURE = true;

        A = TensorDouble.matrix2D(dimension, dimension, 0.0);

        Fq = TensorDouble.matrix2D(size, size, 0.0);

        Wq = TensorDouble.matrix2D(coffSize, size, 0.0);
        Hs = TensorDouble.matrix2D(size, size, 0.0);

        Mcf = TensorDouble.matrix2D(coffSize, coffSize, 0.0);

        ///
        PointUtility.setDiagonalZero(Mcf);

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


/// Tworzymy Macierz vector b vector `b

            Future<DoubleMatrix1D> tensorBTask = executorService.submit(() -> {

                GeometricObject.evaluateForceVector(Fr);                 /// Sily  - F(q)

                Constraint.evaluateConstraintVector(Fi);             /// Wiezy  - Fi(q)

                b.mulitply(-1);

                DoubleMatrix1D matrix1Db = MatrixDoubleUtility.toDenseVector(b);
                return matrix1Db;
            });

/// macierz `A

            Future<DoubleMatrix2D> tensorATask = executorService.submit(() -> {
                /// zeruje macierz A
                A.reset(0.0);

                /// JACOBIAN
                Constraint.getFullJacobian(size, Wq);                     /// Jq = d(Fi)/dq
                TensorDouble WqT = Wq.transposeC();                 /// JqT  - transposedC - copy effective computation
                //WqT = Wq.transpose();                             /// JqT  - transposedC -

                Hs.reset(0.0);

                Constraint.getFullHessian(Hs, SV, size);

                A.setSubMatrix(0, 0, Fq);                   /// procedure SET
                A.plusSubMatrix(0, 0, Hs);                   /// procedure ADD

                A.setSubMatrix(size, 0, Wq);


                A.setSubMatrix(0, size, WqT);

                A.setSubMatrix(size, size, Mcf);

                if (NON_ZERO_CAPTURE) {
                    CaptureCooDoubleMatrix2D captureNonZero = new CaptureCooDoubleMatrix2D(dimension);
                    captureNonZero.forEach(0, 0, Fq);
                    captureNonZero.forEach(0, 0, Hs);        /// --  GPU hessian
                    captureNonZero.forEach(size, 0, Wq);
                    captureNonZero.forEach(0, size, WqT);
                    captureNonZero.forEach(size, size, Mcf);
                    // -------------- stdout -----------------
//                    if (StateReporter.isDebugEnabled()) {
                        captureNonZero.log(StateReporter.getInstance());
//                    }
                }

                DoubleMatrix2D tensorA = MatrixDoubleUtility.toSparse(A);
                return tensorA;
            });

            /*
             *  LU Decomposition  -- Colt Linear Equation Solver
             *
             *   rozwiazujemy zadanie [ A ] * [ dx ] = [ b ]
             */
/// Solver LU Single Iteration Step

            DoubleMatrix2D matrix2DA = null;
            DoubleMatrix1D matrix1Db = null;

            try {
                matrix1Db = tensorBTask.get();

                matrix2DA = tensorATask.get();

            } catch (InterruptedException | ExecutionException e) {
                reporter.writelnf(" [error] tensor calculation pre processing ! %s", e.getMessage());
                e.printStackTrace(System.out);
                solverStat.convergence = false;
                return solverStat;
            }

            if (StateReporter.isDebugEnabled()) {
                reporter.writeln(TensorDouble.writeToString(A));
//                reporter.writeln(TensorDouble.writeToString(b));
            }

            accEvoWatch.stopTick();

            accLUWatch.startTick();

            if (StateReporter.isDebugEnabled()) {
//                reporter.writeln(A.toString(new Integer[0]));
                //reporter.writeln(b.toString(new Integer[0]));
            }

/// LU Solver

            LUDecompositionQuick LU = new LUDecompositionQuick();
            LU.decompose(matrix2DA);

            if (LU.isNonsingular()) {
                LU.solve(matrix1Db);
            } else {
                reporter.writeln("nonsingular : " + LU.isNonsingular());
                solverStat.convergence = false;
                return solverStat;
            }

            accLUWatch.stopTick();

/// Bind delta-x into database points

            dmx = TensorDouble.matrixDoubleFrom(matrix1Db);

            /// uaktualniamy punkty [ SV ] = [ SV ] + [ delta ]
            SV.plus(dmx);
            PointUtility.copyFromStateVector(SV);


            // Lagrange Coefficients
            TensorDouble LC = SV.viewSpan(size, 0, coffSize, 1);

//            if (StateReporter.isDebugEnabled()) {
//                reporter.writeln(TensorDouble.writeToString(LC));
//            }

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

    @Override
    public void destroyDriver() {

        if (executorService == null) {
            return;
        }

        executorService.shutdown();
        try {
            executorService.awaitTermination(1000, TimeUnit.MILLISECONDS);
        } catch (InterruptedException e) {
            executorService = null;
            throw new RuntimeException(e);
        }
    }

}

