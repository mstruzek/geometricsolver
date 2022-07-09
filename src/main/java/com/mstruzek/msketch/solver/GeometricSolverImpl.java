package com.mstruzek.msketch.solver;

import Jama.Matrix;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.LUDecompositionQuick;
import com.mstruzek.msketch.Constraint;
import com.mstruzek.msketch.GeometricPrimitive;
import com.mstruzek.msketch.ParseToColt;
import com.mstruzek.msketch.Point;
import com.mstruzek.msketch.matrix.BindMatrix;
import com.mstruzek.msketch.matrix.MatrixDouble;

import static java.lang.System.out;

public class GeometricSolverImpl implements GeometricSolver {

    // 200x200 macierz A = > 50 okregow/linii  =>   200*200*8 = 320kB

    @Override
    public void solveSystem() {

        out.println("*************************");

        long start = System.currentTimeMillis();                  /// start timing

        if (Point.dbPoint.size() == 0) {
            return;
        }

        /// Tworzymy Macierz "A" - dla tego zadania stala w czasie
        int coffSize = Constraint.allLagrangeCoffSize();
        int size = Point.dbPoint.size() * 2;
        int fullDimension = Point.dbPoint.size() * 2 + coffSize;

        /// TODO  @IniticjalizajaMacierzy

        MatrixDouble A = MatrixDouble.fill(fullDimension, fullDimension, 0.0);
        MatrixDouble Fq = MatrixDouble.fill(size, size, 0.0);
        MatrixDouble Wq = MatrixDouble.fill(coffSize, size, 0.0);

        // right-hand side vector ~ b
        MatrixDouble Fr = MatrixDouble.fill(size, 1, 0.0);
        MatrixDouble Fi = MatrixDouble.fill(coffSize, 1, 0.0);


        Fq = GeometricPrimitive.getAllJacobianForces();

        // Tworzymy wektor prawych stron b
        MatrixDouble b = null;
        BindMatrix dmx = null;

        BindMatrix Bmt = new BindMatrix(Point.dbPoint.size() * 2 + coffSize, 1);
        Bmt.bind(Point.dbPoint);


        double erri1, erri = 0, delta;
        out.println(" Iter/Time [ms] /Norm ");


        //// Matrix -- VIEW on MATRIX

        for (int i = 0; i < 10; i++) {

            /// zerujemy macierz A

            A = MatrixDouble.fill(fullDimension, fullDimension, 0.0);

            /// Tworzymy Macierz vector b

            Fr = GeometricPrimitive.getAllForce();                 /// Sily  - F(q)

            Fi = Constraint.getFullConstraintValues();         /// Wiezy  - Fi(q)

            b = MatrixDouble.mergeByColumn((Fr), (Fi));
            b.dot(-1);

            /// JACOBIAN
            Wq = Constraint.getFullJacobian();                  /// Jq = d(Fi)/dq

            /// HESSIAN
            MatrixDouble Hs = Constraint.getFullHessian(Bmt);

            A.addSubMatrix(0, 0, Fq.addC((Hs))); // TODO NO-COPY???
            A.addSubMatrix(size, 0, Wq);
            A.addSubMatrix(0, size, Wq.transpose());


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
            Fi = Constraint.getFullConstraintValues();

            Matrix normMt = new Matrix(Fi.getArray());
            double norm1 = normMt.norm1();
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

