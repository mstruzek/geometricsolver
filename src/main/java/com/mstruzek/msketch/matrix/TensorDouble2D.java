package com.mstruzek.msketch.matrix;

import com.mstruzek.msketch.Vector;

/**
 * Klasa reprezentuje macierz  dodatkowe operacje , takze w postaci wspoldzielonego widoku !
 */
public class TensorDouble2D implements TensorDouble {

    /*** zmienna przechowujaca nasze elementy */
    public double[][] m = null;
    /*** szerokosc -ilosc kolumn */
    int columns;
    /*** wysokosc - ilosc wiersze */
    int rows;
    /*** base row offset - view concept */
    int rowOffset;
    /***  base column offset - view concept */
    int colOffset;

    /**
     * Konstruktor macierzy
     *
     * @param columns szerokosc -ilosc kolumn
     * @param rows    wysokosc - ilosc wiersze
     */
    public TensorDouble2D(int rows, int columns) {
        this.rowOffset = 0;
        this.colOffset = 0;
        this.columns = columns;
        this.rows = rows;
        this.m = new double[rows][columns];
    }

    /**
     * Shared span with offset into matrix .
     *
     * @param rows
     * @param columns
     */
    public TensorDouble2D(double[][] shared, int rowOffset, int colOffset, int rows, int columns) {
        this.rowOffset = rowOffset;
        this.colOffset = colOffset;
        this.rows = rows;
        this.columns = columns;
        this.m = shared;
    }

    /**
     * Tworzy macierz na podstawie wektora
     *
     * @param vec        wektora
     * @param columnType true if column type, false otherwise row type
     */
    public TensorDouble2D(Vector vec, boolean columnType) {
        this.rowOffset = 0;
        this.colOffset = 0;
        if (columnType) {
            this.columns = 1;
            this.rows = 2;
            this.m = new double[rows][columns];
            this.m[0][0] = vec.getX();
            this.m[1][0] = vec.getY();

        } else {
            this.columns = 2;
            this.rows = 1;
            this.m = new double[rows][columns];
            this.m[0][0] = vec.getX();
            this.m[0][1] = vec.getY();
        }
    }

    @Override
    public double getQuick(int i, int j) {
        return this.m[rowOffset + i][colOffset + j];
    }

    @Override
    public int width() {
        return columns;
    }

    @Override
    public int height() {
        return rows;
    }

    @Override
    public TensorDouble plus(TensorDouble rhs) {
        assertEqualDimensions(this, rhs);
        double value;
        for (int i = 0; i < height(); i++) {
            for (int j = 0; j < width(); j++) {
                this.m[rowOffset + i][colOffset + j] += rhs.getQuick(i, j);
            }
        }
        return this;
    }

    @Override
    public TensorDouble mulitply(double c) {
        for (int i = 0; i < height(); i++) {
            for (int j = 0; j < width(); j++) {
                this.m[rowOffset + i][colOffset + j] *= c;
            }
        }
        return this;
    }

    @Override
    public TensorDouble multiplyC(double c) {
        TensorDouble mt = new TensorDouble2D(width(), height());
        for (int i = 0; i < height(); i++) {
            for (int j = 0; j < width(); j++) {
                mt.setQuick(i, j, mt.getQuick(i, j) * c);
            }
        }
        return mt;
    }


    @Override
    public TensorDouble multiply(TensorDouble rhs) {
        if (this.width() != rhs.height()) throw new Error("Illegal dimension of right-hand side operand matrix");

        if (rhs instanceof SmallTensorDouble && height() == rhs.height() && width() == rhs.width()) {
            double a00 = getQuick(0, 0);
            double a01 = getQuick(0, 1);
            double a10 = getQuick(1, 0);
            double a11 = getQuick(1, 1);
            SmallTensorDouble mt = new SmallTensorDouble(a00, a01, a10, a11);
            return mt.multiply(rhs);
        } else {
            TensorDouble2D mt = new TensorDouble2D(rhs.width(), this.height());
            for (int i = 0; i < height(); i++) { /// this row
                for (int j = 0; j < rhs.width(); j++) { // other column
                    double acc = 0.0;
                    for (int k = 0; k < width(); k++) {     // other column
                        acc += this.getQuick(i, k) * rhs.getQuick(k, j);
                    }
                    mt.setQuick(i, j, acc);
                }
            }
            return mt;
        }
    }

    @Override
    public void setQuick(int r, int c, double value) {
        this.m[rowOffset + r][colOffset + c] = value;
    }

    @Override
    public void plus(int r, int c, double value) {
        this.m[rowOffset + r][colOffset + c] += value;
    }

    @Override
    public TensorDouble viewSpan(int rowOffset, int colOffset, int height, int width) {
        /*
         * first level base tensor sharing policy !
         */
        if(rowOffset + height > height()) {
            throw new Error("out ouf bound offset request");
        }
        if(colOffset + width > width()) {
            throw new Error("out ouf bound offset request");
        }
        TensorDouble2D md = new TensorDouble2D(this.m, rowOffset, colOffset, height, width );
        return md;
    }

    @Override
    public TensorDouble setSubMatrix(int offsetRow, int offsetCol, TensorDouble mt) {
        //sprawdzamy czy macierz wstawiana nie jest za duza
        if (this.height() < (offsetRow + mt.height()) && this.width() < (offsetCol + mt.width())) {
            throw new IllegalStateException("dimension overflow");
        }

        if (mt instanceof ScalarTensorDouble) {
            setQuick(offsetRow, offsetCol, ((ScalarTensorDouble) mt).m);
            return this;
        }
        if (mt instanceof SmallTensorDouble) {
            SmallTensorDouble smt = (SmallTensorDouble) mt;
            setQuick(offsetRow + 0, offsetCol + 0, smt.sm[0]);
            setQuick(offsetRow + 0, offsetCol + 1, smt.sm[1]);
            setQuick(offsetRow + 1, offsetCol + 0, smt.sm[2]);
            setQuick(offsetRow + 1, offsetCol + 1, smt.sm[3]);
            return this;
        }

        TensorDouble2D mtd = (TensorDouble2D) mt;
        /// mozna wstawic
        for (int k = 0; k < mt.height(); k++) {
            System.arraycopy(mtd.m[k], 0, this.m[k + rowOffset + offsetRow], colOffset + offsetCol, mtd.m[k].length);
        }
        return this;
    }

    @Override
    public TensorDouble plusSubMatrix(int offsetRow, int offsetCol, TensorDouble mt) {
        if (this.height() < (offsetRow + mt.height()) || this.width() < (offsetCol + mt.width())) {
            throw new Error("matrix dimension out of bounds");
        }
        if (mt instanceof SmallTensorDouble) {
            SmallTensorDouble smt = (SmallTensorDouble) mt;
            plus(offsetRow + 0, offsetCol + 0, smt.sm[0]);
            plus(offsetRow + 0, offsetCol + 1, smt.sm[1]);
            plus(offsetRow + 1, offsetCol + 0, smt.sm[2]);
            plus(offsetRow + 1, offsetCol + 1, smt.sm[3]);
            return this;
        } else {
            TensorDouble2D tmh = (TensorDouble2D) mt;
            for (int i = 0; i < mt.height(); i++) {
                for (int j = 0; j < mt.width(); j++) {
                    m[i + offsetRow][j + offsetCol] += tmh.m[i][j];
                }
            }
            return this;
        }
    }

    @Override
    public TensorDouble setVector(int r, int c, Vector vector) {
        if (this.width() == 1) { /// row oriented
            setQuick(r + 0, c, vector.getX());
            setQuick(r + 1, c, vector.getY());
            return this;
        } else if (this.height() == 1) { /// column oriented
            setQuick(r, c + 0, vector.getX());
            setQuick(r, c + 1, vector.getY());
            return this;
        } else {
            throw new IllegalStateException("no implementation");
        }
    }

    @Override
    public TensorDouble2D transpose() {
        TensorDouble2D tm = new TensorDouble2D(width(), height());
        for (int i = 0; i < height(); i++) {
            for (int j = 0; j < width(); j++) {
                tm.m[j][i] = this.m[i + rowOffset][j + colOffset];
            }
        }
        return tm;
    }

    @Override
    public TensorDouble reset(double value) {
        for (int i = 0; i < height(); i++)
            for (int j = 0; j < width(); j++)
                m[i + rowOffset][j + colOffset] = value;
        return this;
    }

    @Override
    public <T> T unwrap(Class<T> clazz) {
        if (TensorDouble2D.class.isAssignableFrom(clazz)) {
            return (T) this;
        }
        return null;
    }


    public static void main(String[] args) {
        double q1;
        double m1;
        TensorDouble2D md;
        TensorDouble span;

        md = new TensorDouble2D(8, 8);
        md.setSubMatrix(2, 2, TensorDouble.identity(2, -1.0));
        md.setSubMatrix(4, 2, TensorDouble.rotation(45));
        md.setSubMatrix(6, 2, TensorDouble.rotation(-30));
        md.mulitply(2.0);
        span = md.viewSpan(3, 3, 2, 2);
        span.plus(TensorDouble.identity(2, 200));

        q1 = span.getQuick(0, 0);
        m1 = md.getQuick(3, 3);

        assert m1 == q1;
    }


}
