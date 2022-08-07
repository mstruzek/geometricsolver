package com.mstruzek.msketch.matrix;

import com.mstruzek.msketch.Vector;

public class SmallTensorDouble implements TensorDouble {

    double[] sm = new double[4];

    public SmallTensorDouble() {
    }

    public SmallTensorDouble(double a00, double a01, double a10, double a11) {
        sm[0] = a00;
        sm[1] = a01;
        sm[2] = a10;
        sm[3] = a11;
    }

    @Override
    public double getQuick(int i, int j) {
        return sm[i * 2 + j];
    }

    @Override
    public int width() {
        return 2;
    }

    @Override
    public int height() {
        return 2;
    }

    @Override
    public TensorDouble plus(TensorDouble rhs) {
        SmallTensorDouble mt = new SmallTensorDouble();
        if (rhs instanceof SmallTensorDouble) {
            SmallTensorDouble rh = (SmallTensorDouble) rhs;
            mt.sm[0] = this.sm[0] + rh.sm[0];
            mt.sm[1] = this.sm[1] + rh.sm[1];
            mt.sm[2] = this.sm[2] + rh.sm[2];
            mt.sm[3] = this.sm[3] + rh.sm[3];
            return mt;
        }
        throw new IllegalStateException("no implementation");
    }

    @Override
    public TensorDouble mulitply(double c) {
        this.sm[0] = this.sm[0] * c;
        this.sm[1] = this.sm[1] * c;
        this.sm[2] = this.sm[2] * c;
        this.sm[3] = this.sm[3] * c;
        return this;
    }

    @Override
    public TensorDouble multiplyC(double c) {
        SmallTensorDouble mt = new SmallTensorDouble();
        mt.sm[0] = this.sm[0] * c;
        mt.sm[1] = this.sm[1] * c;
        mt.sm[2] = this.sm[2] * c;
        mt.sm[3] = this.sm[3] * c;
        return mt;
    }

    @Override
    public TensorDouble multiply(TensorDouble rhs) {
        if (rhs instanceof SmallTensorDouble) {
            SmallTensorDouble rh = (SmallTensorDouble) rhs;
            SmallTensorDouble mt = new SmallTensorDouble();
            /// ---------------------------------------
            double a00 = mt.sm[0];
            double a01 = mt.sm[1];
            double a10 = mt.sm[2];
            double a11 = mt.sm[3];
            double b00 = rh.sm[0];
            double b01 = rh.sm[1];
            double b10 = rh.sm[2];
            double b11 = rh.sm[3];
            mt.sm[0] = a00 * b00 + a01 * b10;
            mt.sm[1] = a00 * b01 + a01 * b11;
            mt.sm[2] = a10 * b00 + a11 * b10;
            mt.sm[3] = a10 * b01 + a11 * b11;
            return mt;
        }
        throw new IllegalStateException("invalid dimension");
    }

    @Override
    public void setQuick(int r, int c, double value) {
        sm[r * 2 + c] = value;
    }

    @Override
    public void plus(int r, int c, double value) {
        sm[r * 2 + c] += value;
    }

    @Override
    public TensorDouble viewSpan(int row, int column, int height, int width) {
        throw new IllegalStateException("no implementation");
    }

    @Override
    public TensorDouble setSubMatrix(int offsetRow, int offsetCol, TensorDouble mt) {
        throw new IllegalStateException("no implementation");
    }

    @Override
    public TensorDouble plusSubMatrix(int offsetRow, int offsetCol, TensorDouble mt) {
        throw new IllegalStateException("no implementation");
    }

    @Override
    public TensorDouble setVector(int r, int c, Vector vector) {
        throw new IllegalStateException("no implementation");
    }

    @Override
    public TensorDouble transpose() {
        throw new IllegalStateException("no implementation");
    }

    @Override
    public TensorDouble reset(double value) {
        throw new IllegalStateException("no implementation");
    }

    @Override
    public <T> T unwrap(Class<T> clazz) {
        if(SmallTensorDouble.class.isAssignableFrom(clazz)) {
            return (T) this;
        }
        return null;
    }
}
