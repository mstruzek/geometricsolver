package com.mstruzek.sparsematrixsolver;

import com.mstruzek.msketch.matrix.MatrixDouble;

import java.util.SortedMap;
import java.util.TreeMap;

/**
 * Klasa reprezentuje macierz rzadk�
 *
 * @author root
 */
public class SparseMatrix implements MatrixData {

    /**
     * Kolekcja posortowana - przechowujemy wszystkie elementy
     */
    SortedMap<Index, Double> d = null;

    /**
     * szerokosc
     */
    int width;

    /**
     * wysokosc
     */
    int height;


    public SparseMatrix(int height, int width) {
        super();
        this.width = width;
        this.height = height;
        d = new TreeMap<Index, Double>();
    }

    public SparseMatrix(MatrixDouble md) {
        super();
        this.width = md.width();
        this.height = md.height();
        d = new TreeMap<Index, Double>();
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if (md.get(i, j) != 0.0) d.put(new Index(i, j), md.get(i, j));
            }
        }
    }

    @Override
    public void add(int row, int col, double val) {
        if (d.containsKey(new Index(row, col))) {
            d.put(new Index(row, col), val + d.get(new Index(row, col)).doubleValue());
        }


    }

    @Override
    public double get(int row, int col) {
        if (d.containsKey(new Index(row, col))) {
            return d.get(new Index(row, col)).doubleValue();
        } else
            return 0.0;
    }

    @Override
    public int getHeight() {
        return height;
    }

    @Override
    public int getWidth() {
        return width;
    }

    @Override
    public void multiply(BasicVector out, BasicVector in, int startRow, int startColumn) {

        for (Index id : d.keySet()) {
            out.d[id.getX() + startRow] += d.get(id).doubleValue() * in.d[id.getY() + startColumn];
        }
    }

    @Override
    public void multiply(double in) {
        for (Index id : d.keySet()) {
            d.put(id, d.get(id).doubleValue() * in);
        }
    }

    @Override
    public void set(int row, int col, double val) {
        d.put(new Index(row, col), val);

    }

    public void set(int firstRow, int firstCol, DenseMatrix dm) {

        for (int i = 0; i < dm.getHeight(); i++) {
            for (int j = 0; j < dm.getWidth(); j++) {
                if (dm.d[i][j] != 0.0) {
                    d.put(new Index(firstRow + i, firstCol + j), dm.d[i][j]);
                }
            }
        }
    }

    /**
     * Wstaw macierz rzadka na pozycje
     *
     * @param firstRow
     * @param firstCol
     * @param sm
     */
    public void set(int firstRow, int firstCol, SparseMatrix sm) {

        for (Index id : sm.d.keySet()) {
            d.put(new Index(firstRow + id.getX(), firstCol + id.getY()), sm.d.get(id).doubleValue());
        }
    }

    @Override
    public MatrixData transposeC() {
        SparseMatrix sm = new SparseMatrix(height, width);

        for (Index id : d.keySet()) {
            sm.d.put(id.transposeC(), d.get(id));
        }
        return sm;
    }

    public String toString() {
        String out = new String();
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                out += (d.get(new Index(i, j)) == null ? 0.0 : d.get(new Index(i, j))) + "\t";
            }
            out += "\n";
        }
        return out;

    }

    /**
     * Macierz diagonalna na przekatnej factor
     *
     * @param size
     * @param factor
     * @return
     */
    public static SparseMatrix eye(int size, double factor) {
        SparseMatrix sm = new SparseMatrix(size, size);
        for (int i = 0; i < size; i++) {
            sm.set(i, i, factor);
        }

        return sm;
    }


    public static void main(String[] args) {

        SparseMatrix sm = new SparseMatrix(6, 8);
        sm.set(4, 6, 5);
        sm.set(1, 1, 2);
        sm.set(1, 3, 3);
        //sm.multiply(2);
        //sm = (SparseMatrix) sm.transposeC();

        DenseMatrix dm = DenseMatrix.eye(2, 3.0);

        sm.set(4, 0, dm);


        SparseMatrix sm2 = SparseMatrix.eye(3, 8.0);

        sm.set(0, 4, sm2);
        System.out.println(sm);

        BasicVector bv = new BasicVector(8);
        bv.d[0] = 1;
        bv.d[1] = 1;
        bv.d[2] = 1;
        bv.d[3] = 1;
        bv.d[4] = 1;
        bv.d[5] = 1;
        bv.d[6] = 1;
        bv.d[7] = 1;

        System.out.println(bv);

        BasicVector out = new BasicVector(6);
        sm.multiply(out, bv, 0, 0);
        System.out.println(out);
        System.out.println(sm.d);

        MatrixDouble force = MatrixDouble.matrix2D(5, 5, 0.0);
        force.set(0, 0, 1.0);
        force.set(2, 0, 1.0);
        force.set(0, 2, 1.0);
        SparseMatrix spm = new SparseMatrix(force);
        System.out.println(spm);
        System.out.println(spm.d.size());

    }

}
