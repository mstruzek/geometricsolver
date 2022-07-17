package com.mstruzek.msketch;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import com.mstruzek.msketch.matrix.BindMatrix;
import com.mstruzek.msketch.matrix.MatrixDouble;

/**
 * Klasa zawiera same funkcje static
 * do przeformatowania macierzy gestych na macierze rzadkie
 *
 * @author root
 */
public class ParseToColt {

    /**
     * Konwersja macierzy gestej do macierzy rzadkiej
     *
     * @param md
     * @return
     */
    public static SparseDoubleMatrix2D toSparse(MatrixDouble md) {

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

    public static DoubleMatrix1D toDenseVector(MatrixDouble b) {

        DenseDoubleMatrix1D unwrap = b.unwrap(DenseDoubleMatrix1D.class);
        if(unwrap != null) {
            return unwrap;
        }

        DoubleMatrix1D doubleMatrix1D = new DenseDoubleMatrix1D(b.height());
        for (int i = 0; i < b.height(); i++) {
            doubleMatrix1D.setQuick(i, b.getQuick(i, 0));
        }
        return doubleMatrix1D;
    }

    public static BindMatrix toBindVector(DoubleMatrix1D matrix1Db) {
        BindMatrix dmx = new BindMatrix(matrix1Db.size(), 1);
        for (int i = 0; i < matrix1Db.size(); i++) {
            dmx.set(i, 0, matrix1Db.getQuick(i));
        }
        return dmx;
    }
}
