package com.mstruzek.msketch.matrix;

import cern.colt.matrix.impl.DenseDoubleMatrix1D;

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
    public TensorDouble makeMatrix2D(int rowSize, int colSize, double initValue) {
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
        if (DenseDoubleMatrix1D.class.isAssignableFrom(delegate.getClass())) {
            DenseDoubleMatrix1D denseVector = (DenseDoubleMatrix1D) delegate;
            TensorDouble2D tensorDouble2D = new TensorDouble2D(denseVector.size(), 1);
            for (int i = 0; i < denseVector.size(); i++) {
                tensorDouble2D.setQuick(i, 0, denseVector.getQuick(i));
            }
            return tensorDouble2D;
        }

        throw new Error("not implementation");
    }

}
