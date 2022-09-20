package com.mstruzek.msketch.matrix;

import com.mstruzek.msketch.Vector;

public class ScalarTensorDouble implements TensorDouble {

    double m;

    public ScalarTensorDouble(double value) {
        this.m = value;
    }

    @Override
    public double getQuick(int i, int j) {
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
    public TensorDouble plus(TensorDouble rhs) {
        throw new IllegalStateException("no implementation");
    }

    @Override
    public TensorDouble mulitply(double c) {
        throw new IllegalStateException("no implementation");
    }

    @Override
    public TensorDouble multiplyC(double c) {
        throw new IllegalStateException("no implementation");
    }

    @Override
    public TensorDouble multiply(TensorDouble rhs) {
        throw new IllegalStateException("no implementation");
    }

    @Override
    public void setQuick(int r, int c, double value) {
        throw new IllegalStateException("not implemented");
    }

    @Override
    public void plus(int r, int c, double value) {
        throw new IllegalStateException("not implemented");
    }

    @Override
    public TensorDouble viewSpan(int row, int column, int height, int width) {
        throw new IllegalStateException("no implementation");
    }

    @Override
    public TensorDouble setSubMatrix(int offsetRow, int offsetCol, TensorDouble mt) {
        throw new IllegalStateException("not implemented");
    }

    @Override
    public TensorDouble plusSubMatrix(int offsetRow, int offsetCol, TensorDouble mt) {
        throw new IllegalStateException("not implemented");
    }

    @Override
    public TensorDouble setVector(int r, int c, Vector vector) {
        throw new IllegalStateException("no implementation");
    }

    @Override
    public TensorDouble transpose() {
        throw new IllegalStateException("not implemented");
    }

    @Override
    public TensorDouble transposeC() {
        throw new IllegalStateException("not implemented");
    }

    @Override
    public TensorDouble reset(double value) {
        return null;
    }

    @Override
    public <T> T unwrap(Class<T> clazz) {
        if (ScalarTensorDouble.class.isAssignableFrom(clazz)) {
            return (T) this;
        }
        return null;
    }
}
