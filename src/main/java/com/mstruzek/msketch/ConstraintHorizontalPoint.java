package com.mstruzek.msketch;

import com.mstruzek.msketch.matrix.TensorDouble;

import static com.mstruzek.msketch.ModelRegistry.dbPoint;

/**
 * Klasa reprezentujaca wiez typu polaczenie
 * dwoch punktow(wektorow uogulnionych) PointK[x]-PoinL[x]=0;
 *
 * @author root
 */
public class ConstraintHorizontalPoint extends Constraint {

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
    public ConstraintHorizontalPoint(int constId, Point K, Point L) {
        super(constId, ConstraintType.HorizontalPoint, true);
        //pobierz id
        k_id = K.id;
        l_id = L.id;
    }

    public String toString() {
        double norm = getNorm();
        return "Constraint-Conect2Points" + constraintId + "*s" + size() + " = " + norm + " { K =" + dbPoint.get(k_id) + "  , L = " + dbPoint.get(l_id) + " } \n";
    }

    @Override
    public void getJacobian(TensorDouble mts) {
        TensorDouble mt = mts;
        int j;
        j = po.get(k_id);
        mt.setQuick(0, j * 2, 1.0);         // zero-X

        j = po.get(l_id);
        mt.setQuick(0, j * 2, -1.0);       // zero-X
    }

    @Override
    public boolean isJacobianConstant() {
        return true;
    }

    @Override
    public TensorDouble getValue() {
        final double value = dbPoint.get(k_id).getX() - dbPoint.get(l_id).getX();
        return TensorDouble.scalar(value);
    }

    @Override
    public void getHessian(TensorDouble mt, double lagrange) {
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
        TensorDouble md = getValue();
        return md.getQuick(0, 0);
    }
}
