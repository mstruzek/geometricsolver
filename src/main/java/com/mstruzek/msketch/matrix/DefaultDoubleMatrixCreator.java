package com.mstruzek.msketch.matrix;

import cern.colt.matrix.impl.DenseDoubleMatrix1D;

public class DefaultDoubleMatrixCreator extends MatrixDoubleCreator {

    public static final DefaultDoubleMatrixCreator INSTANCE = new DefaultDoubleMatrixCreator();

    @Override
    public MatrixDouble makeIdentity(int size, double diag) {
        return makeDiagonal(size, diag);
    }

    @Override
    public MatrixDouble makeDiagonal(int size, double diag) {
        if (size < 1) {
            throw new IllegalArgumentException("First argument must be > 0");
        }

        if (size == 2) {
            return new SmallMatrixDouble(diag, 0.0, 0.0, diag);
        }

        MatrixDouble2D I = new MatrixDouble2D(size, size);
        for (int i = 0; i < I.m.length; i++)
            I.m[i][i] = diag;
        return I;
    }

    @Override
    protected MatrixDouble makeDiagonal(double... diag) {
        if (diag.length == 2) {
            double first = diag[0];
            double second = diag[1];
            return new SmallMatrixDouble(first, 0.0, 0.0, second);
        }
        MatrixDouble2D I = new MatrixDouble2D(diag.length, diag.length);
        for (int i = 0; i < I.m.length; i++)
            I.m[i][i] = diag[i];
        return I;
    }

    @Override
    public MatrixDouble makeMatrix2D(int rowSize, int colSize, double initValue) {
        MatrixDouble2D mt = new MatrixDouble2D(rowSize, colSize);
        mt.reset(initValue);
        return mt;
    }

    @Override
    public MatrixDouble makeMatrix1D(int rowSize, double initValue) {
        MatrixDouble2D mt = new MatrixDouble2D(rowSize, 1);
        mt.reset(initValue);
        return mt;
    }

    @Override
    public MatrixDouble matrixDoubleFrom(Object delegate) {
        if (DenseDoubleMatrix1D.class.isAssignableFrom(delegate.getClass())) {
            DenseDoubleMatrix1D denseVector = (DenseDoubleMatrix1D) delegate;
            MatrixDouble2D matrixDouble2D = new MatrixDouble2D(denseVector.size(), 1);
            for (int i = 0; i < denseVector.size(); i++) {
                matrixDouble2D.setQuick(i, 0, denseVector.getQuick(i));
            }
            return matrixDouble2D;
        }

        throw new Error("not implementation");
    }

}
