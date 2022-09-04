package com.mstruzek.msketch;

import com.mstruzek.msketch.matrix.TensorDouble;

import static com.mstruzek.msketch.ModelRegistry.dbPoint;

/**
 * Wiez odpowiedzialny za kat pomiedzy wektorami
 *
 * @author root
 */
public class ConstraintCircleTangency extends Constraint {

    /** Punkty kontrolne */
    /**
     * Point K-id
     */
    int k_id;
    /**
     * Point L-id
     */
    int l_id;
    /**
     * Point M-id
     */
    int m_id;
    /**
     * Point N-id
     */
    int n_id;


    /**
     * Konstruktor pomiedzy 4 punktami prowadzacymi okregi
     * tego wiezu to |(L-K)| + |(N-M)| - |(M-K)| = 0
     *
     * @param K punkt okregu 1 srodkowy
     * @param L punkt okregu 1 promien
     * @param M punkt okregu 2 srodkowy
     * @param N punkt okregu 2 promien
     */
    public ConstraintCircleTangency(Integer constId, Point K, Point L, Point M, Point N) {
        super(constId, ConstraintType.CircleTangency, true);
        k_id = K.id;
        l_id = L.id;
        m_id = M.id;
        n_id = N.id;
    }

    public String toString() {
        double norm = getNorm();
        return "Constraint-CircleTangency" + constraintId + "*s" + size() + " = " + norm + " { K =" + dbPoint.get(k_id) + "  ,L =" + dbPoint.get(l_id) + " ,M =" + dbPoint.get(m_id) + ",N =" + dbPoint.get(n_id) + "} \n";
    }

    @Override
    public TensorDouble getValue() {
        final Vector LK = dbPoint.get(l_id).minus(dbPoint.get(k_id));
        final Vector NM = dbPoint.get(n_id).minus(dbPoint.get(m_id));
        final Vector MK = dbPoint.get(m_id).minus(dbPoint.get(k_id));
        return TensorDouble.scalar(LK.length() + NM.length() - MK.length());
    }

    @Override
    public void getJacobian(TensorDouble mts) {
        TensorDouble mt = mts;
        final Vector vLK = dbPoint.get(l_id).minus(dbPoint.get(k_id)).unit();
        final Vector vNM = dbPoint.get(n_id).minus(dbPoint.get(m_id)).unit();
        final Vector vMK = dbPoint.get(m_id).minus(dbPoint.get(k_id)).unit();
        int j;
        //k
        j = po.get (k_id);
        mt.setVector(0, j * 2, vMK.minus(vLK));
        //l
        j = po.get (l_id);
        mt.setVector(0, j * 2, vLK);
        //m
        j = po.get (m_id);
        mt.setVector(0, j * 2, vMK.product(-1.0).minus(vNM));
        //n
        j = po.get (n_id);
        mt.setVector(0, j * 2, vNM);
    }

    @Override
    public boolean isJacobianConstant() {
        return false;
    }

    @Override
    public void getHessian(TensorDouble mt, double lagrange) {
    }

    @Override
    public boolean isHessianConst() {
        return true;
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
        return m_id;
    }

    @Override
    public int getN() {
        return n_id;
    }

    @Override
    public int getParameter() {
        return -1;
    }

    @Override
    public double getNorm() {
        TensorDouble mt = getValue();
        return mt.getQuick(0, 0);
    }
}
