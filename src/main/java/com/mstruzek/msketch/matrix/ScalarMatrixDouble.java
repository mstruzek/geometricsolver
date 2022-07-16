package com.mstruzek.msketch.matrix;

import com.mstruzek.msketch.Vector;

public class ScalarMatrixDouble implements MatrixDouble {

    double m;

    public ScalarMatrixDouble(double value) {
        this.m = value;
    }

    @Override
    public double get(int i, int j) {
        return m;
    }

    @Override
    public int width() {
        return 1;
    }

    @Override
    public int height() {
        return 1;
    }

    @Override
    public MatrixDouble add(MatrixDouble rhs) {
        throw new IllegalStateException("no implementation");
    }

    @Override
    public MatrixDouble dot(double c) {
        throw new IllegalStateException("no implementation");
    }

    @Override
    public MatrixDouble dotC(double c) {
        throw new IllegalStateException("no implementation");
    }

    @Override
    public MatrixDouble mult(MatrixDouble rhs) {
        throw new IllegalStateException("no implementation");
    }

    @Override
    public MatrixDouble copy() {
        return new ScalarMatrixDouble(this.m);
    }

    @Override
    public void set(int r, int c, double value) {
        throw new IllegalStateException("not implemented");
    }

    @Override
    public void add(int r, int c, double value) {
        throw new IllegalStateException("not implemented");
    }

    @Override
    public MatrixDouble setSubMatrix(int firstRow, int firstColumn, MatrixDouble mt) {
        throw new IllegalStateException("not implemented");
    }

    @Override
    public MatrixDouble addSubMatrix(int firstRow, int firstColumn, MatrixDouble mt) {
        throw new IllegalStateException("not implemented");
    }

    @Override
    public MatrixDouble setVector(int r, int c, Vector vector) {
        throw new IllegalStateException("no implementation");
    }

    @Override
    public MatrixDouble transpose() {
        throw new IllegalStateException("not implemented");
    }

    @Override
    public MatrixDouble reset(double value) {
        return null;
    }
}
