package com.mstruzek.msketch.matrix;

import com.mstruzek.msketch.Vector;

public class SmallMatrixDouble implements MatrixDouble {

    double[] sm = new double[4];

    public SmallMatrixDouble() {
    }

    public SmallMatrixDouble(double a00, double a01, double a10, double a11) {
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
    public MatrixDouble plus(MatrixDouble rhs) {
        SmallMatrixDouble mt = new SmallMatrixDouble();
        if (rhs instanceof SmallMatrixDouble) {
            SmallMatrixDouble rh = (SmallMatrixDouble) rhs;
            mt.sm[0] = this.sm[0] + rh.sm[0];
            mt.sm[1] = this.sm[1] + rh.sm[1];
            mt.sm[2] = this.sm[2] + rh.sm[2];
            mt.sm[3] = this.sm[3] + rh.sm[3];
            return mt;
        }
        throw new IllegalStateException("no implementation");
    }

    @Override
    public MatrixDouble mulitply(double c) {
        this.sm[0] = this.sm[0] * c;
        this.sm[1] = this.sm[1] * c;
        this.sm[2] = this.sm[2] * c;
        this.sm[3] = this.sm[3] * c;
        return this;
    }

    @Override
    public MatrixDouble multiplyC(double c) {
        SmallMatrixDouble mt = new SmallMatrixDouble();
        mt.sm[0] = this.sm[0] * c;
        mt.sm[1] = this.sm[1] * c;
        mt.sm[2] = this.sm[2] * c;
        mt.sm[3] = this.sm[3] * c;
        return mt;
    }

    @Override
    public MatrixDouble multiply(MatrixDouble rhs) {
        if (rhs instanceof SmallMatrixDouble) {
            SmallMatrixDouble rh = (SmallMatrixDouble) rhs;
            SmallMatrixDouble mt = new SmallMatrixDouble();
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
    public MatrixDouble viewSpan(int row, int column, int height, int width) {
        throw new IllegalStateException("no implementation");
    }

    @Override
    public MatrixDouble setSubMatrix(int offsetRow, int offsetCol, MatrixDouble mt) {
        throw new IllegalStateException("no implementation");
    }

    @Override
    public MatrixDouble plusSubMatrix(int offsetRow, int offsetCol, MatrixDouble mt) {
        throw new IllegalStateException("no implementation");
    }

    @Override
    public MatrixDouble setVector(int r, int c, Vector vector) {
        throw new IllegalStateException("no implementation");
    }

    @Override
    public MatrixDouble transpose() {
        throw new IllegalStateException("no implementation");
    }

    @Override
    public MatrixDouble reset(double value) {
        throw new IllegalStateException("no implementation");
    }

    @Override
    public <T> T unwrap(Class<T> clazz) {
        if(SmallMatrixDouble.class.isAssignableFrom(clazz)) {
            return (T) this;
        }
        return null;
    }
}
