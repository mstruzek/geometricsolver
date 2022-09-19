package com.mstruzek.msketch.matrix;

public abstract class MatrixDoubleCreator {

    private static MatrixDoubleCreator creator = DefaultDoubleMatrixCreator.INSTANCE;

    public static MatrixDoubleCreator getInstance() {
        return creator;
    }

    public static void setInstance(MatrixDoubleCreator creator) {
        MatrixDoubleCreator.creator = creator;
    }

    /**
     * Make usually small identity matrix 2x2.
     *
     * @param size an integer > 0.
     * @return size x size two dimensional MatrixDouble.
     */
    public abstract TensorDouble makeIdentity(int size, double diag);

    /**
     * Make small diagonal matrix usually 2x2 .
     *
     * @param size dimension
     * @param diag initial value
     * @return
     */
    public abstract TensorDouble makeDiagonal(int size, double diag);

    /**
     * Make small matrix usually 2x2 with all values zero and values on main diagonal .
     *
     * @param values values
     * @return
     */
    protected abstract TensorDouble makeDiagonal(double... values);

    /**
     * Standard matrix two dimensional , rowSize X colSize.
     *
     * @param rowSize
     * @param colSize
     * @param initValue
     * @param capture
     * @return
     */
    public abstract TensorDouble makeMatrix2D(int rowSize, int colSize, double initValue, boolean capture);

    /**
     * Column oriented matrix one dimensional - vector.
     *
     * @param rowSize
     * @param initValue
     * @return
     */
    public abstract TensorDouble makeMatrix1D(int rowSize, double initValue);

    /**
     * Create column oriented one dimensional matrix backed by usually delegate shared DoubleMatrix1D.
     * Do not copy if possible.
     * @param delegate matrix
     * @return adapter
     */
    public abstract TensorDouble matrixDoubleFrom(Object delegate);

}
