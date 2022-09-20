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
    public TensorDouble makeIdentity(int size, double diag) {
        throw new Error("not implemented");
    }

    @Override
    public TensorDouble makeDiagonal(int size, double diag) {
        throw new Error("not implemented");
    }

    @Override
    protected TensorDouble makeDiagonal(double... values) {
        throw new Error("no implementation found for small tensors");
    }

    @Override
    public TensorDouble makeMatrix2D(int rowSize, int colSize, double initValue) {
        TensorDouble matrix2D = new SparseDoubleTensor2DImpl(rowSize, colSize);
        if (initValue != 0.0) {
            matrix2D.reset(initValue);
        }
        return matrix2D;
    }

    @Override
    public TensorDouble makeMatrix1D(int rows, double initValue) {
        DenseDoubleTensor1DImpl matrix1D = new DenseDoubleTensor1DImpl(rows);
        if (initValue != 0.0) {
            matrix1D.reset(initValue);
        }
        return matrix1D;
    }

    @Override
    public TensorDouble matrixDoubleFrom(Object delegate) {
        if (DenseDoubleMatrix1D.class.isAssignableFrom(delegate.getClass())) {
            DenseDoubleMatrix1D denseDoubleMatrix1D = (DenseDoubleMatrix1D) delegate;
            TensorDouble tensorDouble = new DenseDoubleTensor1DImpl(denseDoubleMatrix1D);
            return tensorDouble;
        }
        throw new Error("not implemented");
    }


    /**
     * Dominantly used for Columnar Storage - Column Vector.
     */
    public static class DenseDoubleTensor1DImpl implements TensorDouble {

        private final DoubleMatrix1D mt;

        private DenseDoubleTensor1DImpl(DoubleMatrix1D copyOrView) {
            this.mt = copyOrView;
        }

        public DenseDoubleTensor1DImpl(int size) {
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
        public TensorDouble plus(TensorDouble rhs) {
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
        public TensorDouble mulitply(double c) {
            this.mt.assign(new DoubleFunction() {
                @Override
                public double apply(double value) {
                    return value * c;
                }
            });
            return this;
        }

        @Override
        public TensorDouble multiplyC(double c) {
            DoubleMatrix1D denseCopy = this.mt.copy();
            DenseDoubleTensor1DImpl denseDoubleMatrix1D = new DenseDoubleTensor1DImpl(denseCopy);
            denseDoubleMatrix1D.mulitply(c);
            return denseDoubleMatrix1D;
        }

        @Override
        public TensorDouble multiply(TensorDouble rhs) {
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
        public void plus(int r, int c, double value) {
            if (c != 0) {
                throw new Error("columnar vector, out of bound subscript use");
            }
            mt.setQuick(r, mt.getQuick(r) + value);
        }

        @Override
        public TensorDouble viewSpan(int row, int column, int height, int width) {
            if (column != 0) {
                throw new Error("columnar vector, out of bound subscript use");
            }
            if (width != 1) {
                throw new Error("columnar vector, out of bound subscript use");
            }
            return new DenseDoubleTensor1DImpl(this.mt.viewPart(row, height));
        }

        @Override
        public TensorDouble setSubMatrix(int offsetRow, int offsetCol, TensorDouble mt) {
            if (offsetCol != 0) {
                throw new Error("columnar vector, out of bound subscript use");
            }

            TensorDouble2D matrix2d = mt.unwrap(TensorDouble2D.class);
            if (matrix2d != null) {

                double a00 = matrix2d.m[0][0];
                double a10 = matrix2d.m[1][0];

                this.mt.setQuick(offsetRow, a00);
                this.mt.setQuick(offsetRow + 1, a10);

                return this;
            }

            ScalarTensorDouble scalar = mt.unwrap(ScalarTensorDouble.class);
            if (scalar != null) {

                this.mt.setQuick(offsetRow, scalar.m);

                return this;
            }


            DoubleMatrix1D doubleMatrix1D = mt.unwrap(DoubleMatrix1D.class);
            if (doubleMatrix1D != null) {
                DoubleMatrix1D viewPart = this.mt.viewPart(offsetRow, mt.height());

                viewPart.assign(doubleMatrix1D);

                return this;
            }

            throw new Error("not implemented");
        }

        @Override
        public TensorDouble plusSubMatrix(int offsetRow, int offsetCol, TensorDouble mt) {
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
        public TensorDouble setVector(int r, int c, Vector vector) {
            if (c != 0) {
                throw new Error("columnar vector, out of bound subscript use");
            }
            mt.setQuick(r, vector.getX());
            mt.setQuick(r + 1, vector.getY());
            return this;
        }

        @Override
        public TensorDouble transpose() {
            throw new Error("not implemented");
        }

        @Override
        public TensorDouble reset(double value) {
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

    public static class SparseDoubleTensor2DImpl implements TensorDouble {

        private DoubleMatrix2D mt;

        // column intention or row intention depends on mt dimension
        boolean colIntention;

        private SparseDoubleTensor2DImpl(DoubleMatrix2D view) {
            this.mt = view;
            this.colIntention = this.mt.rows() > this.mt.columns();
        }

        public SparseDoubleTensor2DImpl(int rows, int columns) {
            this.mt = new SparseDoubleMatrix2D(rows, columns);
            this.colIntention = this.mt.rows() > this.mt.columns();
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
        public TensorDouble plus(TensorDouble rhs) {
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
        public TensorDouble mulitply(double c) {
            this.mt.forEachNonZero(new IntIntDoubleFunction() {
                @Override
                public double apply(int i, int j, double value) {
                    return value * c;
                }
            });
            return this;
        }

        @Override
        public TensorDouble multiplyC(double c) {
            throw new Error("not implemented");
        }

        @Override
        public TensorDouble multiply(TensorDouble rhs) {
            throw new Error("not implemented");
        }

        @Override
        public void setQuick(int r, int c, double value) {
            this.mt.setQuick(r, c, value);
        }

        @Override
        public void plus(int r, int c, double value) {
            this.mt.setQuick(r, c, this.mt.getQuick(r, c) + value);
        }

        @Override
        public TensorDouble viewSpan(int row, int column, int height, int width) {
            DoubleMatrix2D view = this.mt.viewPart(row, column, height, width);
            return new SparseDoubleTensor2DImpl(view);
        }

        @Override
        public TensorDouble setSubMatrix(int offsetRow, int offsetCol, TensorDouble mt) {

            SmallTensorDouble smd = mt.unwrap(SmallTensorDouble.class);
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

            SparseDoubleMatrix2D doubleMatrix2D = mt.unwrap(SparseDoubleMatrix2D.class);
            if (null != doubleMatrix2D) {
                DoubleMatrix2D viewPart = this.mt.viewPart(offsetRow, offsetCol, doubleMatrix2D.rows(), doubleMatrix2D.columns());
                viewPart.assign(doubleMatrix2D);
                return this;
            }

            throw new Error("not implemented");
        }

        @Override
        public TensorDouble plusSubMatrix(int offsetRow, int offsetCol, TensorDouble mt) {

            SmallTensorDouble smd = mt.unwrap(SmallTensorDouble.class);
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

            throw new Error("not implemented");
        }

        @Override
        public TensorDouble setVector(int r, int c, Vector vector) {
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
        public TensorDouble transpose() {
            return new SparseDoubleTensor2DImpl(this.mt.viewDice());
        }

        @Override
        public TensorDouble reset(double value) {
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
