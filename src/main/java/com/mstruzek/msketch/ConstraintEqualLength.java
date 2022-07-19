package com.mstruzek.msketch;

import com.mstruzek.msketch.matrix.MatrixDouble;

import static com.mstruzek.msketch.Point.dbPoint;

/**
 * Klasa reprezentujaca wiez rownej dlugosci pomiedzy
 * dwoma wektorami (4 punkty)
 *
 * @author root
 */
public class ConstraintEqualLength extends Constraint {

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
     * Konstruktor pomiedzy 4 punktami lub
     * 2 punktami i FixLine(czyli 2 wektory)
     * rownanie tego wiezu to sqrt[(K-L)'*(K-L)] - sqrt[(M-N)'*(M-N)] = 0
     * iloczyn skalarny
     *
     * @param constId
     * @param K
     * @param L
     * @param M
     * @param N
     */
    public ConstraintEqualLength(int constId, Point K, Point L, Point M, Point N) {
        this(constId, K, L, M, N, true);
    }

    public ConstraintEqualLength(int constId, Point K, Point L, Point M, Point N, boolean persistent) {
        super(constId, GeometricConstraintType.EqualLength, persistent);
        k_id = K.id;
        l_id = L.id;
        m_id = M.id;
        n_id = N.id;
    }

    public String toString() {
        double norm = getNorm();
        return "Constraint-LinesSameLength" + constraintId + "*s" + size() + " = " + norm + " { K =" + dbPoint.get(k_id) + "  ,L =" + dbPoint.get(l_id) + " ,M =" + dbPoint.get(m_id) + ",N =" + dbPoint.get(n_id) + "} \n";
    }

    @Override
    public void getJacobian(MatrixDouble mts) {
        Vector LK = dbPoint.get(l_id).sub(dbPoint.get(k_id)).unit();
        Vector NM = dbPoint.get(n_id).sub(dbPoint.get(m_id)).unit();
        MatrixDouble mt = mts;
        int j;
        //k
        j = po.get(k_id);
        mt.setVector(0, j * 2, LK.dot(-1.0));
        //l
        j = po.get(l_id);
        mt.setVector(0, j * 2, LK);
        //m
        j = po.get(m_id);
        mt.setVector(0, j * 2, NM);
        //n
        j = po.get(n_id);
        mt.setVector(0, j * 2, NM.dot(-1.0));
    }

    @Override
    public boolean isJacobianConstant() {
        return true;
    }

    @Override
    public MatrixDouble getValue() {
        Vector LK = dbPoint.get(l_id).sub(dbPoint.get(k_id));
        Vector NM = dbPoint.get(n_id).sub(dbPoint.get(m_id));
        double value = LK.length() - NM.length();
        return MatrixDouble.scalar(value);
    }

    @Override
    public MatrixDouble getHessian(double lagrange) {
        return null;
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
        MatrixDouble md = getValue();
        return md.getQuick(0, 0);
    }

}
