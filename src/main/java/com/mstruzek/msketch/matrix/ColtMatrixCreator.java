package com.mstruzek.msketch.matrix;

public class ColtMatrixCreator extends MatrixDoubleCreator {

    @Override
    public MatrixDouble makeRotation2d(double alfa) {
        throw new IllegalStateException("it is not implemented for to small matrix");
    }

    @Override
    public MatrixDouble makeIdentity(int size, double diag) {
        throw new IllegalStateException("it is not implemented for to small matrix");
    }

    @Override
    public MatrixDouble makeDiagonal(int size, double diag) {
        throw new IllegalStateException("it is not implemented for to small matrix");
    }

    @Override
    protected MatrixDouble makeDiagonal(double... values) {
        throw new IllegalStateException("it is not implemented for to small matrix");
    }

    @Override
    public MatrixDouble makeMatrix2D(int rowSize, int colSize, double initValue) {
        return null;
    }

    @Override
    public MatrixDouble makeMatrix1D(int colums, double initValue) {
        return null;
    }

    @Override
    public MatrixDouble makeMatrix1Dtr(int rows, double initValue) {
        return null;
    }
}
