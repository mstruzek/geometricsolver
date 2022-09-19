package com.mstruzek.msketch.matrix;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

public class DefaultDoubleMatrixCreator extends MatrixDoubleCreator {

    public static final DefaultDoubleMatrixCreator INSTANCE = new DefaultDoubleMatrixCreator();

    @Override
    public TensorDouble makeIdentity(int size, double diag) {
        return makeDiagonal(size, diag);
    }

    @Override
    public TensorDouble makeDiagonal(int size, double diag) {
        if (size < 1) {
            throw new IllegalArgumentException("First argument must be > 0");
        }

        if (size == 2) {
            return new SmallTensorDouble(diag, 0.0, 0.0, diag);
        }

        TensorDouble2D I = new TensorDouble2D(size, size);
        for (int i = 0; i < I.m.length; i++)
            I.m[i][i] = diag;
        return I;
    }

    @Override
    protected TensorDouble makeDiagonal(double... diag) {
        if (diag.length == 2) {
            double first = diag[0];
            double second = diag[1];
            return new SmallTensorDouble(first, 0.0, 0.0, second);
        }
        TensorDouble2D I = new TensorDouble2D(diag.length, diag.length);
        for (int i = 0; i < I.m.length; i++)
            I.m[i][i] = diag[i];
        return I;
    }

    @Override
    public TensorDouble makeMatrix2D(int rowSize, int colSize, double initValue, boolean capture) {
        TensorDouble2D mt = new TensorDouble2D(rowSize, colSize);
        mt.reset(initValue);
        return mt;
    }

    @Override
    public TensorDouble makeMatrix1D(int rowSize, double initValue) {
        TensorDouble2D mt = new TensorDouble2D(rowSize, 1);
        mt.reset(initValue);
        return mt;
    }

    @Override
    public TensorDouble matrixDoubleFrom(Object delegate) {
        if (DoubleMatrix1D.class.isAssignableFrom(delegate.getClass())) {
            DoubleMatrix1D denseVector = (DoubleMatrix1D) delegate;
            TensorDouble2D tensorDouble2D = new TensorDouble2D(denseVector.size(), 1);
            forEachNonZero(denseVector, (i, value) -> {
                tensorDouble2D.setQuick(i, 0, value);
            });
            return tensorDouble2D;
        }

        if(DoubleMatrix2D.class.isAssignableFrom(delegate.getClass())) {
            DoubleMatrix2D denseMatrix = (DoubleMatrix2D) delegate;
            final TensorDouble2D tensorDouble2D = new TensorDouble2D(denseMatrix.rows(), denseMatrix.columns());
            forEachNonZero(denseMatrix, (r, c, value) -> {
                tensorDouble2D.setQuick(r, c, value);
            });

            return tensorDouble2D;

        }

        throw new Error("not implementation");
    }


    public interface IntIntDoubleFunction {
        void apply(int var1, int var2, double value);
    }

    public interface IntDoubleFunction {
        void apply(int var1, double value);
    }

    static void forEachNonZero(DoubleMatrix2D denseMatrix, IntIntDoubleFunction var1 )  {
        int var2 = denseMatrix.rows();

        while(true) {
            --var2;
            if (var2 < 0) {
                return;
            }

            int var3 = denseMatrix.columns();

            while(true) {
                --var3;
                if (var3 < 0) {
                    break;
                }

                double var4 = denseMatrix.getQuick(var2, var3);
                if (var4 != 0.0) {
                    var1.apply(var2, var3, var4);
                }
            }
        }
    }


    static void forEachNonZero(DoubleMatrix1D denseVector, IntDoubleFunction var1 ) {
        int var2 = denseVector.size();

        while (true) {
            --var2;
            if (var2 < 0) {
                return;
            }

            double var4 = denseVector.getQuick(var2);
            if (var4 != 0.0) {
                var1.apply(var2, var4);
            }
        }
    }
}
