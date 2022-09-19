package com.mstruzek.msketch.matrix;

import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import com.mstruzek.msketch.Vector;
import com.mstruzek.msketch.solver.StateReporter;

import java.util.Objects;
import java.util.TreeMap;


/**
 * Capture values in COO format into SoA and report to stdout.
 */
public class CaptureCooDoubleMatrix2D implements TensorDouble {

    private int idx = 0;


    private static class TensorCoo implements Comparable<TensorCoo> {
        public final int i;
        public final int i1;

        public TensorCoo(int i, int i1) {
            this.i = i;
            this.i1 = i1;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            TensorCoo tensorCoo = (TensorCoo) o;
            return i == tensorCoo.i && i1 == tensorCoo.i1;
        }

        @Override
        public int hashCode() {
            return Objects.hash(i, i1);
        }

        public static TensorCoo of(int i, int i1) {
            return new TensorCoo(i, i1);
        }

        @Override
        public int compareTo(TensorCoo o) {
            if (i > o.i) return 1;
            else if (i == o.i && i1 > o.i1) return 1;
            else if (i == o.i && i1 == o.i1) return 0;
            return -1;
        }
    }

    private TreeMap<TensorCoo, Double> tensorCooValue = new TreeMap<>();

    TensorDouble delegate;

    public CaptureCooDoubleMatrix2D(TensorDouble mt) {
        super();
        this.delegate = mt;
        this.idx = 0;
    }

    public void log(StateReporter reporter) {
        reporter.writelnf("------ COO format , nnz ( %d )  [ %d , %d ]", idx, this.delegate.width(), this.delegate.height());
        for (var entry : tensorCooValue.entrySet()) {
            reporter.writelnf("( %d , %d ) , %e ", entry.getKey().i, entry.getKey().i1, entry.getValue());
        }
        reporter.writelnf("------");
    }


    @Override
    public int width() {
        return this.delegate.width();
    }

    @Override
    public int height() {
        return this.delegate.height();
    }

    @Override
    public double getQuick(int i, int i1) {
        throw new IllegalStateException("only tensor A or view");
    }

    @Override
    public TensorDouble plus(TensorDouble rhs) {
        throw new IllegalStateException("only tensor A or view");
    }

    @Override
    public TensorDouble mulitply(double c) {
        throw new IllegalStateException("only tensor A or view");
    }

    @Override
    public TensorDouble multiplyC(double c) {
        throw new IllegalStateException("only tensor A or view");
    }

    @Override
    public TensorDouble multiply(TensorDouble rhs) {
        throw new IllegalStateException("only tensor A or view");
    }

    @Override
    public void setQuick(int i, int i1, double v) {
        throw new IllegalStateException("only tensor A or view");
    }

    @Override
    public void plus(int r, int c, double value) {
        throw new IllegalStateException("only tensor A or view");
    }

    @Override
    public TensorDouble viewSpan(int row, int column, int height, int width) {
        throw new IllegalStateException("only tensor A or view");
    }

    @Override
    public TensorDouble setSubMatrix(int offsetRow, int offsetCol, TensorDouble mt) {
        SparseDoubleMatrix2D unwrap = mt.unwrap(SparseDoubleMatrix2D.class);
        if (unwrap == null) {
            throw new IllegalStateException("expected sparse double tensor !");
        }
        unwrap.forEachNonZero((i, i1, v) -> {
            int row = offsetRow + i;
            int column = offsetCol + i1;
            tensorCooValue.put(TensorCoo.of(row, column), v);
            return v;
        });
        this.delegate.setSubMatrix(offsetRow, offsetCol, mt);
        return this;
    }

    @Override
    public TensorDouble plusSubMatrix(int offsetRow, int offsetCol, TensorDouble mt) {
        SparseDoubleMatrix2D unwrap = mt.unwrap(SparseDoubleMatrix2D.class);
        if (unwrap == null) {
            throw new IllegalStateException("expected sparse double tensor !");
        }
        unwrap.forEachNonZero((i, i1, v) -> {
            TensorCoo coo = TensorCoo.of(offsetRow + i, offsetCol + i1);
            Double value = tensorCooValue.get(coo);
            if (value == null) {
                value = v;
            }
            tensorCooValue.put(coo, value);
            return v;
        });
        this.delegate.plusSubMatrix(offsetRow, offsetCol, mt);
        return this;
    }

    @Override
    public TensorDouble setVector(int r, int c, Vector vector) {
        throw new IllegalStateException("only tensor A or view");
    }

    @Override
    public TensorDouble transpose() {
        throw new IllegalStateException("only tensor A or view");
    }

    @Override
    public TensorDouble reset(double value) {
        this.idx = 0;
        tensorCooValue.clear();
        this.delegate.reset(value);
        return this;
    }

    @Override
    public <T> T unwrap(Class<T> clazz) {
        if (CaptureCooDoubleMatrix2D.class.isAssignableFrom(clazz)) {
            return (T) this;
        }
        return this.delegate.unwrap(clazz);
    }
}
