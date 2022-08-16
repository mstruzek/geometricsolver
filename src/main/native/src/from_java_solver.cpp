/**
 *  /// wydzielic pliku CU
 *
 *  /// przygotowac petle
 *
 *  ///
 *
 */
void solveSystemOnGPUAA(int *error) {

        int size;      /// wektor stanu
        int coffSize;  /// wspolczynniki Lagrange
        int dimension; /// dimension = size + coffSize

        solverWatch.reset();
        accEvoWatch.reset();
        accLUWatch.reset();

        pointOffset = stateOffset(points, [](auto point) { return point->id; });

        accConstraintSize = accumalatedValue(constraints, graph::constraintSize);

        accGeometricSize = accumalatedValue(geometrics, graph::geometricSetSize);

        size = std::accumulate(geometrics.begin(), geometrics.end(), 0, [](auto acc, auto const& geometric) { return acc + graph::geometricSetSize(geometric); });

        coffSize =
            std::accumulate(constraints.begin(), constraints.end(), 0, [](auto acc, auto const& constraint) { return acc + graph::constraintSize(constraint); });

        dimension = size + coffSize;

        
        /// Uklad rownan liniowych  [ A * x = b ] powstały z linerazycji ukladu dynamicznego

        /// 1# CPU -> GPU inicjalizacja macierzy

        ///------------------------
        graph::Tensor A;  /// Macierz głowna ukladu rownan liniowych
        graph::Tensor Fq; /// Macierz sztywnosci ukladu obiektow zawieszonych na sprezynach.
        graph::Tensor Wq; /// d(FI)/dq - Jacobian Wiezow

        /// HESSIAN
        graph::Tensor Hs;

        // Wektor prawych stron [Fr; Fi]'
        graph::Tensor b;

        // skladowe to Fr - oddzialywania pomiedzy obiektami, sily w sprezynach
        graph::Tensor Fr;

        // skladowa to Fiq - wartosci poszczegolnych wiezow
        graph::Tensor Fi;

        double norm1;            /// wartosci bledow na wiezach
        double prevNorm;         /// norma z wczesniejszej iteracji,
        double errorFluctuation; /// fluktuacja bledu

        stat.startTime = graph::TimeNanosecondsNow();

        printf("@#=================== Solver Initialized ===================#@ \n");
        printf("");

        if (points.size() == 0) {
                printf("[warning] - empty model\n");
                return;
        }

        if (constraints.size()) {
                printf("[warning] - no constraint configuration applied \n");
                return;
        }

        /// Tworzymy Macierz "A" - dla tego zadania stala w czasie

        size = points.size() * 2;
        coffSize = AllLagrangeCoffSize();
        dimension = size + coffSize;

        stat.size = size;
        stat.coefficientArity = coffSize;
        stat.dimension = dimension;

        accEvoWatch.setStartTick();

        /// Inicjalizacje bazowych macierzy rzadkich - SparseMatrix

        /// --->
        // instead loop over vector space provide static access to point location in VS.
        // - reference locations for Jacobian and Hessian evaluation !
        PointLocationSetup();

        /// cMalloc() --- przepisac na GRID -->wspolrzedne

        using graph::Tensor;

        A; //= Tensor.tensor2D(dimension, dimension, 0.0);

        Fq; // = Tensor.tensor2D(size, size, 0.0);

        Wq; // = Tensor.tensor2D(coffSize, size, 0.0);
        Hs; // = Tensor.tensor2D(size, size, 0.0);

        /// KERNEL_O

        /// ### macierz sztywnosci stala w czasie - jesli bezkosztowe bliskie zero to zawieramy w KELNER_PRE
        /// ( dla uproszczenia tylko w pierwszy przejsciu -- Thread_0_Grid_0 - synchronization Guard )

        /// inicjalizacja i bezposrednio do A
        ///   w nastepnych przejsciach -> kopiujemy

        // KernelEvaluateStiffnessMatrix<<<>>>(Fq)

        /// cMallloc

        /// Wektor prawych stron b

        b; /// ---  = Tensor.matrix1D(dimension, 0.0);

        // right-hand side vector ~ b

        Fr; // = b.viewSpan(0, 0, size, 1);
        Fi; // = b.viewSpan(size, 0, coffSize, 1);

        Tensor dmx;

        /// State Vector - zmienne stanu
        Tensor SV; // = Tensor.matrix1D(dimension, 0.0);

        /// PointUtility.

        CopyIntoStateVector(NULL);  // SV
        SetupLagrangeMultipliers(); // SV

        accEvoWatch.setStopTick();

        norm1 = 0.0;
        prevNorm = 0.0;
        errorFluctuation = 0.0;

        int itr = 0;
        while (itr < MAX_SOLVER_ITERATIONS) {

                accEvoWatch.setStartTick();

                /// --- KERNEL_PRE

                /// zerujemy macierz A

                /// # KERNEL_PRE

                A; // .reset(0.0); /// --- ze wzgledu na addytywnosc

                /// Tworzymy Macierz vector b vector `b

                /// # KERNEL_PRE
                GeometricObjectEvaluateForceVector(); // Fr /// Sily  - F(q)
                ConstraintEvaluateConstraintVector(); // Fi / Wiezy  - Fi(q)

                // b.setSubMatrix(0,0, (Fr));
                // b.setSubMatrix(size,0, (Fi));

                b; // .mulitply(-1);

                /// macierz `A

                /// # KERNEL_PRE (__shared__ JACOBIAN)

                ConstraintGetFullJacobian(); // --- (Wq); /// Jq = d(Fi)/dq --- write without intermediary matrix   `A = `A set value

                Hs; // --- .reset(0.0); /// ---- niepotrzebnie

                /// # KERNEL_PRE

                ConstraintGetFullHessian(); // --- (Hs, SV, size); // --- write without intermediate matrix '`A = `A +value

                A.setSubTensor(0, 0, Fq);  /// procedure SET
                A.plusSubTensor(0, 0, Hs); /// procedure ADD

                A.setSubTensor(size, 0, Wq);
                A.setSubTensor(0, size, Wq.transpose());

                /*
                 *  LU Decomposition  -- Colt Linear Equation Solver
                 *
                 *   rozwiazjemy zadanie [ A ] * [ dx ] = [ b ]
                 */

                /// DENSE MATRIX - pierwsze podejscie !

                /// ---- KERNEL_PRE

                accEvoWatch.setStopTick();

                accLUWatch.setStartTick();

                /// DENSE - CuSolver
                /// LU Solver

                /// ---------------- < cuSOLVER >
                /// ---------------- < cuSOLVER >
                /// ---------------- < cuSOLVER >
                // LUDecompositionQuick LU = new LUDecompositionQuick();
                // LU.decompose(matrix2DA);
                // LU.solve(matrix1Db);

                accLUWatch.setStopTick();

                /// Bind delta-x into database points

                /// --- KERNEL_POST
                dmx; // --- = Tensor.matrixDoubleFrom(matrix1Db);

                /// uaktualniamy punkty [ SV ] = [ SV ] + [ delta ]

                /// --- KERNEL_POST
                SV.plus(dmx);

                PointUtilityCopyFromStateVector(); // ---(SV); /// ??????? - niepotrzebnie , NA_KONCU

                /// --- KERNEL_POST( __synchronize__ REDUCTION )

                norm1 = ConstraintGetFullNorm();

                /// cCopySymbol()

                printf(" [ step :: %02d]  duration [ns] = %,12d  norm = %e \n", itr, (accLUWatch.stopTick - accEvoWatch.startTick), norm1);

                /// Gdy po 5-6 przejsciach iteracji, normy wiezow kieruja sie w strone minimum energii, to repozycjonowac prowadzace punkty

                if (norm1 < CONVERGENCE_LIMIT) {
                        stat.error = norm1;
                        printf("fast convergence - norm [ %e ] \n", norm1);
                        printf("constraint error = %e \n", norm1);
                        printf("");
                        break;
                }

                /// liczymy zmiane bledu
                errorFluctuation = norm1 - prevNorm;
                prevNorm = norm1;
                stat.error = norm1;

                if (itr > 1 && errorFluctuation / prevNorm > 0.70) {
                        printf("CHANGES - STOP ITERATION *******");
                        printf(" errorFluctuation          : %d \n", errorFluctuation);
                        printf(" relative error            : %f \n", (errorFluctuation / norm1));
                        solverWatch.setStopTick();
                        stat.constraintDelta = ConstraintGetFullNorm();
                        stat.convergence = false;
                        stat.stopTime = graph::TimeNanosecondsNow();
                        stat.iterations = itr;
                        stat.accSolverTime = accLUWatch.accTime;
                        stat.accEvaluationTime = accEvoWatch.accTime;
                        stat.timeDelta = solverWatch.stopTick - solverWatch.startTick;
                        return;
                }
                itr++;
        }

        solverWatch.setStopTick();
        long solutionDelta = solverWatch.delta();

        printf("# solution delta : %d \n", solutionDelta); // print execution time
        printf("\n");                                      // print execution time

        stat.constraintDelta = ConstraintGetFullNorm();
        stat.convergence = norm1 < CONVERGENCE_LIMIT;
        stat.stopTime = graph::TimeNanosecondsNow();
        stat.iterations = itr;
        stat.accSolverTime = accLUWatch.accTime;
        stat.accEvaluationTime = accEvoWatch.accTime;
        stat.timeDelta = solverWatch.stopTick - solverWatch.startTick;

        return;
}