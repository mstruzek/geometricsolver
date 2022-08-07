package com.mstruzek.msketch;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import com.mstruzek.msketch.matrix.TensorDouble;

/**
 * Klasa zawiera same funkcje static  do przeformatowania macierzy gestych na macierze rzadkie
 *
 * @author root
 */
public class MatrixDoubleUtility {

    /**
     * jesli mozliwe pomijamy kopiowanie przy konwersja do macierzy rzadkiej.
     *
     * @param md
     * @return
     */
    public static SparseDoubleMatrix2D toSparse(TensorDouble md) {

        SparseDoubleMatrix2D unwrap = md.unwrap(SparseDoubleMatrix2D.class);
        if(unwrap != null) {
            return unwrap;
        }

        SparseDoubleMatrix2D matrix2D = new SparseDoubleMatrix2D(md.height(), md.width());

        for (int i = 0; i < md.height(); i++) {
            for (int j = 0; j < md.width(); j++) {
                if (md.getQuick(i, j) != 0.0) {
                    matrix2D.setQuick(i, j, md.getQuick(i, j));
                }
            }
        }
        return matrix2D;
    }

    /**
     * Jesli mozliwe pomijamy kopiowanie przy konwersji do DoubleMatrix1D.
     * @param md matrix double
     * @return
     */
    public static DoubleMatrix1D toDenseVector(TensorDouble md) {

        DenseDoubleMatrix1D unwrap = md.unwrap(DenseDoubleMatrix1D.class);
        if(unwrap != null) {
            return unwrap;
        }

        DoubleMatrix1D doubleMatrix1D = new DenseDoubleMatrix1D(md.height());
        for (int i = 0; i < md.height(); i++) {
            doubleMatrix1D.setQuick(i, md.getQuick(i, 0));
        }
        return doubleMatrix1D;
    }

}
