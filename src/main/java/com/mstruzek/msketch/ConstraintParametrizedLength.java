package com.mstruzek.msketch;

import com.mstruzek.msketch.matrix.TensorDouble;

import static com.mstruzek.msketch.ModelRegistry.dbPoint;

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
    public void getJacobian(TensorDouble mts) {
        TensorDouble mt = mts;
        Vector LK = dbPoint.get(l_id).minus(dbPoint.get(k_id));
        Vector NM = dbPoint.get(n_id).minus(dbPoint.get(m_id));
        double lk = LK.length();
        double nm = NM.length();
        double d = (param_id != -1) ? ModelRegistry.dbParameter.get(param_id).getValue() : 1.0;
        int j;

        //k
        j = po.get(k_id);
        mt.setVector(0, j * 2, LK.product(-1.0 * d / lk));

        //l
        j = po.get(l_id);
        mt.setVector(0, j * 2, LK.product(1.0 * d / lk));

        //m
        j = po.get(m_id);
        mt.setVector(0, j * 2, NM.product(1.0 / nm));

        //n
        j = po.get(n_id);
        mt.setVector(0, j * 2, NM.product(-1.0 / nm));
    }

    @Override
    public boolean isJacobianConstant() {
        return false;
    }

    @Override
    public TensorDouble getValue() {
        Vector LK = dbPoint.get(l_id).minus(dbPoint.get(k_id));
        Vector NM = dbPoint.get(n_id).minus(dbPoint.get(m_id));
        double d = (param_id != -1) ? ModelRegistry.dbParameter.get(param_id).getValue() : 1.0;
        double value = d * LK.length() - NM.length();
        return TensorDouble.scalar(value);
    }

    @Override
    public TensorDouble getHessian(double lagrange) {
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
        TensorDouble md = getValue();
        return md.getQuick(0, 0);
    }
}
