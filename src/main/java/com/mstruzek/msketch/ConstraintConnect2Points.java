package com.mstruzek.msketch;

import com.mstruzek.msketch.matrix.MatrixDouble;
import com.mstruzek.msketch.matrix.MatrixDouble2D;

import static com.mstruzek.msketch.ModelRegistry.dbPoint;

/**
 * Klasa reprezentujaca wiez typu polaczenie
 * dwoch punktow(wektorow uogulnionych) PointK-PoinL=0;
 *
 * @author root
 */
public class ConstraintConnect2Points extends Constraint {

    /**
     * Point K-id
     */
    int k_id;
    /**
     * Point L-id
     */
    int l_id;

    /**
     * Konstruktor wiezu
     *
     * @param constId
     * @param K       - zrzutowany Point na Vector
     * @param L       - Vector w ktorym bedzie zafiksowany K
     */
    public ConstraintConnect2Points(int constId, Point K, Point L) {
        super(constId, GeometricConstraintType.Connect2Points, true);
        k_id = K.id;
        l_id = L.id;
    }

    public String toString() {
        double norm = getNorm();
        return "Constraint-Conect2Points" + constraintId + "*s" + size() + " = " + norm + " { K =" + dbPoint.get(k_id) + "  , L = " + dbPoint.get(l_id) + " } \n";
    }

    @Override
    public void getJacobian(MatrixDouble mts) {
        MatrixDouble mt = mts;
        int j = 0;
        //k
        j = po.get(k_id);
        mt.setSubMatrix(0, j * 2, MatrixDouble.identity(2, 1.0));        //macierz jednostkowa = I

        //l
        j = po.get(l_id);
        mt.setSubMatrix(0, j * 2, MatrixDouble.identity(2, -1.0));       // = -I
    }

    @Override
    public boolean isJacobianConstant() {
        return true;
    }

    @Override
    public MatrixDouble getValue() {
        return new MatrixDouble2D(dbPoint.get(k_id).Vector().minus(dbPoint.get(l_id)), true);
    }

    @Override
    public MatrixDouble getHessian(double lagrange) {
        return null;
    }

    @Override
    public boolean isHessianConst() {
        return false;
    }

    @Override
    public int getK() {
        return k_id;
    }

    @Override
    public int getL() {
        return l_id;
    }

    @Override
    public int getM() {
        return -1;
    }

    @Override
    public int getN() {
        return -1;
    }

    @Override
    public int getParameter() {
        return -1;
    }

    @Override
    public double getNorm() {
        MatrixDouble md = getValue();
        return Math.sqrt(md.getQuick(0, 0) * md.getQuick(0, 0) + md.getQuick(1, 0) * md.getQuick(1, 0));
    }

}
