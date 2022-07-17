package com.mstruzek.msketch.matrix;

import cern.colt.function.DoubleDoubleFunction;
import cern.colt.function.DoubleFunction;
import cern.colt.function.IntIntDoubleFunction;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import com.mstruzek.msketch.Vector;

/**
 * Heavy Matrix Creator for SparseDoubleMatrix2D and DenseVector1D !
 *
 */
final public class ColtMatrixCreator extends MatrixDoubleCreator {

    public static ColtMatrixCreator INSTANCE = new ColtMatrixCreator();

    @Override
    public MatrixDouble makeIdentity(int size, double diag) {
        throw new Error("not implemented");
    }

    @Override
    public MatrixDouble makeDiagonal(int size, double diag) {
        throw new Error("not implemented");
    }

    @Override
    protected MatrixDouble makeDiagonal(double... values) {
        throw new Error("no implementation found for small tensors");
    }

    @Override
    public MatrixDouble makeMatrix2D(int rowSize, int colSize, double initValue) {
        SparseDoubleMatrix2DImpl matrix2D = new SparseDoubleMatrix2DImpl(rowSize, colSize);
        if (initValue != 0.0) {
            matrix2D.reset(initValue);
        }
        return matrix2D;
    }

    @Override
    public MatrixDouble makeMatrix1D(int rows, double initValue) {
        DenseDoubleMatrix1DImpl matrix1D = new DenseDoubleMatrix1DImpl(rows);
        if (initValue != 0.0) {
            matrix1D.reset(initValue);
        }
        return matrix1D;
    }

    @Override
    public MatrixDouble matrixDoubleFrom(Object delegate) {
        if (DenseDoubleMatrix1D.class.isAssignableFrom(delegate.getClass())) {
            DenseDoubleMatrix1D denseDoubleMatrix1D = (DenseDoubleMatrix1D) delegate;
            MatrixDouble matrixDouble = new DenseDoubleMatrix1DImpl(denseDoubleMatrix1D);
            return matrixDouble;
        }
        throw new Error("not implemented");
    }

    /*
     *  ##SparseDoubleMatrix2D
     *  public void set(int row, int column, double value) {
     *
     *  public abstract void setQuick(int row, int column, double value);
     *
     *  public double get(int row, int column)
     *
     *  public abstract double getQuick(int row, int column);
     *  public DoubleMatrix2D assign(double[][] values) {       Function.Add. Function.Multiply
     *
     *  public DoubleMatrix1D viewColumn(int column) {
     *
     *  public DoubleMatrix1D viewRow(int row) {
     *
     *  public DoubleMatrix2D viewDice() {                                                  TRANSPOSE
     *
     *  public DoubleMatrix2D viewSelection(int[] rowIndexes, int[] columnIndexes) {
     *
     *  public DoubleMatrix2D viewPart(int row, int column, int height, int width) {        SUB-MATRIX
     *
     *  public DoubleMatrix2D copy() {
     *
     *
     *
     * Conversion from vector:  Matrix1D  =>  Matrix2D
     *
     *              # Jacobian evaluation !
     *
     *              [ Matrix2D ] .viewColumn ( int column ). assign( [ Matrix1D ] )
     *
     *  public SparseDoubleMatrix2D(double[][] var1) {
     *
     */


    /**
     * Dominantly used for Columnar Storage - Column Vector.
     */
    public static class DenseDoubleMatrix1DImpl implements MatrixDouble {

        private final DoubleMatrix1D mt;

        private DenseDoubleMatrix1DImpl(DoubleMatrix1D copyOrView) {
            this.mt = copyOrView;
        }

        public DenseDoubleMatrix1DImpl(int size) {
            this.mt = new DenseDoubleMatrix1D(size);
        }

        @Override
        public int width() {
            return 1;
        }

        @Override
        public int height() {
            return mt.size();
        }

        @Override
        public double getQuick(int i, int j) {
            if (j != 0) {
                throw new Error("columnar vector, out of bound subscript use");
            }
            double value = mt.getQuick(i);
            return value;
        }

        @Override
        public MatrixDouble add(MatrixDouble rhs) {
            DenseDoubleMatrix1D doubleMatrix1D = rhs.unwrap(DenseDoubleMatrix1D.class);
            if(doubleMatrix1D != null) {
                this.mt.assign(doubleMatrix1D, new DoubleDoubleFunction() {
                    @Override
                    public double apply(double v, double v1) {
                        return v + v1;
                    }
                });
                return this;
            }
            throw new Error("not implemented");
        }

        @Override
        public MatrixDouble dot(double c) {
            this.mt.assign(new DoubleFunction() {
                @Override
                public double apply(double value) {
                    return value * c;
                }
            });
            return this;
        }

        @Override
        public MatrixDouble dotC(double c) {
            DoubleMatrix1D denseCopy = this.mt.copy();
            DenseDoubleMatrix1DImpl denseDoubleMatrix1D = new DenseDoubleMatrix1DImpl(denseCopy);
            denseDoubleMatrix1D.dot(c);
            return denseDoubleMatrix1D;
        }

        @Override
        public MatrixDouble mult(MatrixDouble rhs) {
            throw new Error("not implemented");
        }

        @Override
        public void setQuick(int r, int c, double value) {
            if (c != 0) {
                throw new Error("columnar vector, out of bound subscript use");
            }
            mt.setQuick(r, value);
        }

        @Override
        public void add(int r, int c, double value) {
            if (c != 0) {
                throw new Error("columnar vector, out of bound subscript use");
            }
            mt.setQuick(r, mt.getQuick(r) + value);
        }

        @Override
        public MatrixDouble viewSpan(int row, int column, int height, int width) {
            if (column != 0) {
                throw new Error("columnar vector, out of bound subscript use");
            }
            if (width != 1) {
                throw new Error("columnar vector, out of bound subscript use");
            }
            return new DenseDoubleMatrix1DImpl(this.mt.viewPart(row, height));
        }

        @Override
        public MatrixDouble setSubMatrix(int offsetRow, int offsetCol, MatrixDouble mt) {
            if (offsetCol != 0) {
                throw new Error("columnar vector, out of bound subscript use");
            }

            DoubleMatrix1D doubleMatrix1D = mt.unwrap(DoubleMatrix1D.class);
            if (doubleMatrix1D != null) {
                DoubleMatrix1D viewPart = this.mt.viewPart(offsetRow, mt.height());

                viewPart.assign(doubleMatrix1D);

                return this;
            }

            ScalarMatrixDouble scalar = mt.unwrap(ScalarMatrixDouble.class);
            if (scalar != null) {

                this.mt.setQuick(offsetRow, scalar.m);

                return this;
            }

            MatrixDouble2D matrix2d = mt.unwrap(MatrixDouble2D.class);
            if (matrix2d != null) {

                double a00 = matrix2d.m[0][0];
                double a10 = matrix2d.m[1][0];

                this.mt.setQuick(offsetRow, a00);
                this.mt.setQuick(offsetRow + 1, a10);

                return this;
            }

            throw new Error("not implemented");
        }

        @Override
        public MatrixDouble addSubMatrix(int offsetRow, int offsetCol, MatrixDouble mt) {
            if (offsetCol != 0) {
                throw new Error("columnar vector, out of bound subscript use");
            }
            DoubleMatrix1D doubleMatrix1D = mt.unwrap(DoubleMatrix1D.class);
            if (doubleMatrix1D == null) {
                throw new Error("not implemented");
            }

            DoubleMatrix1D viewPart = this.mt.viewPart(offsetRow, mt.height());

            viewPart.assign(doubleMatrix1D, new DoubleDoubleFunction() {
                @Override
                public double apply(double v, double v1) {
                    return v + v1;
                }
            });
            return this;
        }

        @Override
        public MatrixDouble setVector(int r, int c, Vector vector) {
            if (c != 0) {
                throw new Error("columnar vector, out of bound subscript use");
            }
            mt.setQuick(r, vector.getX());
            mt.setQuick(r + 1, vector.getY());
            return this;
        }

        @Override
        public MatrixDouble transpose() {
            throw new Error("not implemented");
        }

        @Override
        public MatrixDouble reset(double value) {
            this.mt.assign(value);
            return this;
        }

        @Override
        public <T> T unwrap(Class<T> clazz) {
            if (DoubleMatrix1D.class.isAssignableFrom(clazz)) {
                return (T) this.mt;
            }
            return null;
        }
    }

    public static class SparseDoubleMatrix2DImpl implements MatrixDouble {

        private DoubleMatrix2D mt;

        private SparseDoubleMatrix2DImpl(DoubleMatrix2D view) {
            this.mt = view;
        }

        public SparseDoubleMatrix2DImpl(int rows, int columns) {
            this.mt = new SparseDoubleMatrix2D(rows, columns);
        }

        @Override
        public int width() {
            return mt.columns();
        }

        @Override
        public int height() {
            return mt.rows();
        }

        @Override
        public double getQuick(int i, int j) {
            return mt.getQuick(i, j);
        }

        @Override
        public MatrixDouble add(MatrixDouble rhs) {
            SparseDoubleMatrix2D doubleMatrix2D = rhs.unwrap(SparseDoubleMatrix2D.class);

            if (doubleMatrix2D == null) {
                throw new Error("not implemented");
            }

            this.mt.assign(doubleMatrix2D, new DoubleDoubleFunction() {
                @Override
                public double apply(double v, double v1) {
                    return v + v1;      /// v = v + v1
                }
            });
            return this;
        }

        @Override
        public MatrixDouble dot(double c) {
            this.mt.forEachNonZero(new IntIntDoubleFunction() {
                @Override
                public double apply(int i, int j, double value) {
                    return value * c;
                }
            });
            return this;
        }

        @Override
        public MatrixDouble dotC(double c) {
            throw new Error("not implemented");
        }

        @Override
        public MatrixDouble mult(MatrixDouble rhs) {
            throw new Error("not implemented");
        }

        @Override
        public void setQuick(int r, int c, double value) {
            this.mt.setQuick(r, c, value);
        }

        @Override
        public void add(int r, int c, double value) {
            this.mt.setQuick(r, c, this.mt.getQuick(r, c) + value);
        }

        @Override
        public MatrixDouble viewSpan(int row, int column, int height, int width) {
            DoubleMatrix2D view = this.mt.viewPart(row, column, height, width);
            return new SparseDoubleMatrix2DImpl(view);
        }

        @Override
        public MatrixDouble setSubMatrix(int offsetRow, int offsetCol, MatrixDouble mt) {

            SparseDoubleMatrix2D doubleMatrix2D = mt.unwrap(SparseDoubleMatrix2D.class);
            if (null != doubleMatrix2D) {
                DoubleMatrix2D viewPart = this.mt.viewPart(offsetRow, offsetCol, doubleMatrix2D.rows(), doubleMatrix2D.columns());
                viewPart.assign(doubleMatrix2D);
                return this;
            }

            SmallMatrixDouble smd = mt.unwrap(SmallMatrixDouble.class);
            if (smd != null) {
                DoubleMatrix2D viewPart = this.mt.viewPart(offsetRow, offsetCol, mt.height(), mt.width());
                double a00 = smd.sm[0];
                double a01 = smd.sm[1];
                double a10 = smd.sm[2];
                double a11 = smd.sm[3];
                viewPart.setQuick(0, 0, a00);
                viewPart.setQuick(0, 1, a01);
                viewPart.setQuick(1, 0, a10);
                viewPart.setQuick(1, 1, a11);
                return this;
            }

            throw new Error("not implemented");
        }

        @Override
        public MatrixDouble addSubMatrix(int offsetRow, int offsetCol, MatrixDouble mt) {

            SparseDoubleMatrix2D doubleMatrix2D = mt.unwrap(SparseDoubleMatrix2D.class);
            if (doubleMatrix2D != null) {
                DoubleMatrix2D viewPart = this.mt.viewPart(offsetRow, offsetCol, doubleMatrix2D.rows(), doubleMatrix2D.columns());
                viewPart.assign(doubleMatrix2D, new DoubleDoubleFunction() {
                    @Override
                    public double apply(double v, double v1) {
                        return v + v1;                              /// v = v + v1
                    }
                });
                return this;
            }

            SmallMatrixDouble smd = mt.unwrap(SmallMatrixDouble.class);
            if (smd != null) {
                DoubleMatrix2D viewPart = this.mt.viewPart(offsetRow, offsetCol, mt.height(), mt.width());
                double a00 = smd.sm[0];
                double a01 = smd.sm[1];
                double a10 = smd.sm[2];
                double a11 = smd.sm[3];
                viewPart.setQuick(0, 0, viewPart.getQuick(0, 0) + a00);
                viewPart.setQuick(0, 1, viewPart.getQuick(0, 1) + a01);
                viewPart.setQuick(1, 0, viewPart.getQuick(1, 0) + a10);
                viewPart.setQuick(1, 1, viewPart.getQuick(1, 1) + a11);
                return this;
            }

            throw new Error("not implemented");
        }

        @Override
        public MatrixDouble setVector(int r, int c, Vector vector) {

            ///  !(this.mt.rows() == 1 || this.mt.rows() == 2 ) &&
            boolean colIntention = this.mt.rows() > this.mt.columns();

            if (colIntention) {
                this.mt.setQuick(r + 0, c, vector.getX());
                this.mt.setQuick(r + 1, c, vector.getY());
            } else {
                this.mt.setQuick(r, c + 0, vector.getX());
                this.mt.setQuick(r, c + 1, vector.getY());
            }
            return this;
        }

        @Override
        public MatrixDouble transpose() {
            return new SparseDoubleMatrix2DImpl(this.mt.viewDice());
        }

        @Override
        public MatrixDouble reset(double value) {
            /* light-weight matrix for re-instantiation in-place */
            this.mt = new SparseDoubleMatrix2D(this.height(), this.width());
            if (value != 0) {
                throw new Error("always reset sparse matrix to zero");
            }
            return this;
        }

        @Override
        public <T> T unwrap(Class<T> clazz) {
            if (SparseDoubleMatrix2D.class.isAssignableFrom(clazz)) {
                return (T) this.mt;
            }
            return null;
        }
    }

}
