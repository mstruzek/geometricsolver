package com.mstruzek.msketch.matrix;

public abstract class MatrixDoubleCreator {

    private static MatrixDoubleCreator creator;

    public static MatrixDoubleCreator getCreatorInstance() {
        if (creator == null) {
            creator = new DefaultDoubleMatrixCreator();
        }
        return creator;
    }


    public abstract MatrixDouble makeRotation2d(double alfa);

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
     * @param colSize
     * @param initValue
     * @return
     */
    public abstract MatrixDouble makeMatrix1D(int colSize, double initValue);

    /**
     * Row oriented matrix one dimensional - transpose vector.
     *
     * @param rowSize
     * @param initValue
     * @return
     */
    public abstract MatrixDouble makeMatrix1Dtr(int rowSize, double initValue);

}
