package com.mstruzek.msketch.solver;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.LUDecompositionQuick;
import com.mstruzek.msketch.Constraint;
import com.mstruzek.msketch.GeometricPrimitive;
import com.mstruzek.msketch.ParseToColt;
import com.mstruzek.msketch.matrix.BindMatrix;
import com.mstruzek.msketch.matrix.MatrixDouble;

import static com.mstruzek.msketch.Constraint.allLagrangeCoffSize;
import static com.mstruzek.msketch.Point.dbPoint;
import static java.lang.System.out;

public class GeometricSolverImpl implements GeometricSolver {

    // 200x200 macierz A = > 50 okregow/linii  =>   200*200*8 = 320kB

    @Override
    public void solveSystem() {

        out.println("*************************");

        long start = System.currentTimeMillis();                  /// start timing

        if (dbPoint.size() == 0) {
            return;
        }

        /// Tworzymy Macierz "A" - dla tego zadania stala w czasie

        /// TODO  @IniticjalizajaMacierzy

        final int size = dbPoint.size() * 2;
        final int coffSize = allLagrangeCoffSize();
        final int dimension = size + coffSize;

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


        double erri1, erri = 0, delta;
        out.println(" Iter/Time [ms] /Norm ");


        //// Matrix -- VIEW on MATRIX


        //// ADD ----> SET ( reuse mts )

        for (int i = 0; i < 10; i++) {

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


            /**
             *  LU Decomposition  -- Colt Linera Equatio Solver
             *
             */

            // rozwiazjemy zadanie A*dx=b

            /// Zaproponowac Wspolny interfejs Macierzowy

            DoubleMatrix2D matrix2DA = ParseToColt.toSparse(A);
            DoubleMatrix1D matrix1Db = ParseToColt.toDenseVector(b);

            long stepStart = System.currentTimeMillis(); // start timing
            {
                LUDecompositionQuick LU = new LUDecompositionQuick();
                LU.decompose(matrix2DA);
                LU.solve(matrix1Db);
            }
            // stop timing
            long stepDelta = System.currentTimeMillis() - stepStart;

            dmx = ParseToColt.toBindVector(matrix1Db);


            /// uaktualniamy punkty
            Bmt.plusEquals(dmx);
            Bmt.copyToPoints();


            /// AFTER -- copyToPoints  x2
            Constraint.getFullConstraintValues(Fi);

            double norm1 = Fi.norm1();
            out.println(" \n " + (i + 1) + " || " + stepDelta + "  ||  " + norm1 + "\n");

            //stary warunek wyjscia
            if (norm1 < 0.05) {
                double constraintNorm = Constraint.getFullNorm();
                out.println("New Norm + :" + constraintNorm);
                break;
            }

            if (i == 0) {
                erri = norm1;
            }

            //liczymy zmiane bledu
            if (i > 0) {
                erri1 = norm1;
                delta = erri1 - erri;
                erri = erri1;
                if (delta > 0) {
                    out.println("CHANGES - STOP ITERATION *******\n");
                    return;
                }
            }
        }
        long solutionDelta = System.currentTimeMillis() - start;
        out.println("solution delta [ ms ] : " + solutionDelta); // print execution time
    }

}

