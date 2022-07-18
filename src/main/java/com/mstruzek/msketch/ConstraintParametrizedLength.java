package com.mstruzek.msketch;

import com.mstruzek.msketch.matrix.MatrixDouble;

import static com.mstruzek.msketch.Point.dbPoint;

/**
 * Klasa reprezentujaca wiez wspolczynnika wagi dlugosci pomiedzy
 * dwoma wektorami (4 punkty)
 *
 * @author root
 */
public class ConstraintParametrizedLength extends Constraint {
    // Punkty kontrolne
    /*** Point K-id */
    int k_id;
    /*** Point L-id */
    int l_id;
    /*** Point M-id */
    int m_id;
    /*** Point N-id */
    int n_id;
    /*** Numer parametru przechowujacy wartosc proporcjonalnosci */
    int param_id;

    /**
     * Konstruktor pomiedzy 4 punktami lub
     * 2 punktami i FixLine(czyli 2 wektory)
     * rownanie tego wiezu to  d * sqrt[(K-L)'*(K-L)] - sqrt[(M-N)'*(M-N)] = 0
     * iloczyn skalarny
     *
     * @param constId
     * @param K
     * @param L
     * @param M
     * @param N
     * @param parameter
     */
    public ConstraintParametrizedLength(int constId, Point K, Point L, Point M, Point N, Parameter parameter) {
        this(constId, K, L, M, N, parameter, true);
    }

    public ConstraintParametrizedLength(int constId, Point K, Point L, Point M, Point N, Parameter parameter, boolean persistent) {
        super(constId, GeometricConstraintType.ParametrizedLength, persistent);
        k_id = K.id;
        l_id = L.id;
        m_id = M.id;
        n_id = N.id;
        param_id = (parameter != null) ? parameter.getId() : -1;
    }

    public String toString() {
        double norm = getNorm();
        return "Constraint-EqualLengthG" + constraintId + "*s" + size() + " = " + norm + " { K =" + dbPoint.get(k_id) + "  ,L =" + dbPoint.get(l_id) + " ,M =" + dbPoint.get(m_id) + ",N =" + dbPoint.get(n_id) + "} \n";
    }

    @Override
    public void getJacobian(MatrixDouble mts) {
        Vector LK = dbPoint.get(l_id).sub(dbPoint.get(k_id));
        Vector NM = dbPoint.get(n_id).sub(dbPoint.get(m_id));
        double lk = LK.length();
        double nm = NM.length();
        double d = (param_id != -1) ? Parameter.dbParameter.get(param_id).getValue() : 1.0;
        int j = 0;
        for (Integer i : dbPoint.keySet()) {
            if (k_id == dbPoint.get(i).id) {
                mts.setVector(0, j * 2, LK.dot(-1.0 * d / lk));
            }
            if (l_id == dbPoint.get(i).id) {
                mts.setVector(0, j * 2, LK.dot(1.0 * d / lk));
            }
            if (m_id == dbPoint.get(i).id) {
                mts.setVector(0, j * 2, NM.dot(1.0 / nm));
            }
            if (n_id == dbPoint.get(i).id) {
                mts.setVector(0, j * 2, NM.dot(-1.0 / nm));
            }
            j++;
        }
    }

    @Override
    public boolean isJacobianConstant() {
        return false;
    }

    @Override
    public MatrixDouble getValue() {
        Vector LK = dbPoint.get(l_id).sub(dbPoint.get(k_id));
        Vector NM = dbPoint.get(n_id).sub(dbPoint.get(m_id));
        double d = (param_id != -1) ? Parameter.dbParameter.get(param_id).getValue() : 1.0;
        double value = d * LK.length() - NM.length();
        return MatrixDouble.scalar(value);
    }

    @Override
    public MatrixDouble getHessian(double lagrange) {
        /// bez Hessianu zbierznosc ponizje 1e-10 przy pierwszej iteracji
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
        return param_id;
    }

    @Override
    public double getNorm() {
        MatrixDouble md = getValue();
        return md.getQuick(0, 0);
    }
}
