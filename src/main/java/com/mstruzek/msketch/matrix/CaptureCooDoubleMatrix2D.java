package com.mstruzek.msketch.matrix;

import cern.colt.function.IntIntDoubleFunction;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import com.mstruzek.msketch.solver.StateReporter;

import java.util.Comparator;
import java.util.Objects;
import java.util.TreeMap;

/**
 * Capture values in COO format into SoA and report to stdout.
 */
public class CaptureCooDoubleMatrix2D {

    private TreeMap<TensorCoo, Double> tensorCooValue = new TreeMap<>();

    final int width;

    public CaptureCooDoubleMatrix2D(int width) {
        this.width = width;
    }

    public void forEach(int r, int c, TensorDouble tensor) {
        SparseDoubleMatrix2D unwrap = tensor.unwrap(SparseDoubleMatrix2D.class);
        if (unwrap == null)
            return;

        unwrap.forEachNonZero(new IntIntDoubleFunction() {
            @Override
            public double apply(int i, int i1, double v) {
                TensorCoo key = new TensorCoo(r + i, c + i1);
                if (tensorCooValue.get(key) != null) throw new IllegalStateException("key exists ! " + key);
                tensorCooValue.put(key, v);
                return v;
            }
        });
    }

    /**
     * Treat each element of a tensor as transposed matrix.
     * @param r      output tensor row offset
     * @param c      output tensor column offset
     * @param tensor input tensor
     */
    public void forEachTranspose(int r, int c, TensorDouble tensor) {
        SparseDoubleMatrix2D unwrap = tensor.unwrap(SparseDoubleMatrix2D.class);
        if (unwrap == null)
            return;

        unwrap.forEachNonZero(new IntIntDoubleFunction() {
            @Override
            public double apply(int i, int i1, double v) {
                TensorCoo key = new TensorCoo(r + i1, c + i);
                if (tensorCooValue.get(key) != null) throw new IllegalStateException("key exists ! " + key);
                tensorCooValue.put(key, v);
                return v;
            }
        });
    }


    public void log(StateReporter reporter) {
        reporter.writelnf("------ COO format* , ");
        int i = 0;
        for (var entry : tensorCooValue.entrySet()) {
            reporter.writelnf("%3d , %d  %d - %7.3f ", i, entry.getKey().i, entry.getKey().i1, entry.getValue());
            i++;
        }
        reporter.writelnf("------");
    }

    private class TensorCoo implements Comparable<TensorCoo> {
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

        public TensorCoo of(int i, int i1) {
            return new TensorCoo(i, i1);
        }

        @Override
        public int compareTo(TensorCoo other) {
            return Comparator.<TensorCoo>comparingInt(coo -> coo.i * width + coo.i1)
                .compare(this, other);
        }
    }

}
