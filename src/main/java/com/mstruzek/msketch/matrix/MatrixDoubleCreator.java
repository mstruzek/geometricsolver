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
    public abstract MatrixDouble makeIdentity(int size, double diag);

    /**
     * Make small diagonal matrix usually 2x2 .
     *
     * @param size dimension
     * @param diag initial value
     * @return
     */
    public abstract MatrixDouble makeDiagonal(int size, double diag);

    /**
     * Make small matrix usually 2x2 with all values zero and values on main diagonal .
     *
     * @param values values
     * @return
     */
    protected abstract MatrixDouble makeDiagonal(double... values);

    /**
     * Standard matrix two dimensional , rowSize X colSize.
     *
     * @param rowSize
     * @param colSize
     * @param initValue
     * @return
     */
    public abstract MatrixDouble makeMatrix2D(int rowSize, int colSize, double initValue);

    /**
     * Column oriented matrix one dimensional - vector.
     *
     * @param rowSize
     * @param initValue
     * @return
     */
    public abstract MatrixDouble makeMatrix1D(int rowSize, double initValue);

}
