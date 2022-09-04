package com.mstruzek.msketch;

import com.mstruzek.msketch.matrix.TensorDouble;

import static com.mstruzek.msketch.ModelRegistry.dbParameter;
import static com.mstruzek.msketch.ModelRegistry.dbPoint;

/**
 * Wiez odpowiedzialny za kat pomiedzy wektorami
 *
 * @author root
 */
public class ConstraintAngle2Lines extends Constraint {

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
     * Numer parametru przechowujacy kat w radianach
     */
    int param_id;


    /**
     * Konstruktor pomiedzy 4 punktami i paramtetrem
     * rownanie tego wiezu to (L-K)'*(N-M)-cos(param)*|L-K|*|N-M| = 0
     *
     * @param K punkt prostej
     * @param L punkt prostej
     * @param M punkt prostej
     * @param N punkt prostej
     */
    public ConstraintAngle2Lines(Integer constId, Point K, Point L, Point M, Point N, Parameter param) {
        super(constId, ConstraintType.Angle2Lines, true);
        k_id = K.id;
        l_id = L.id;
        m_id = M.id;
        n_id = N.id;
        param_id = param.getId();
    }

    public String toString() {
        double norm = getNorm();
        return "Constraint-Angle2Lines" + constraintId + "*s" + size() + " = " + norm +
            " { K =" + dbPoint.get(k_id) + "  ,L =" + dbPoint.get(l_id) + " ,M =" + dbPoint.get(m_id) + ",N =" + dbPoint.get(n_id) +
            ", Parametr-" + dbParameter.get(param_id).getId() + " = " + dbParameter.get(param_id).getValue() + "} \n";
    }

    @Override
    public TensorDouble getValue() {
        final Vector LK = dbPoint.get(l_id).minus(dbPoint.get(k_id));
        final Vector NM = dbPoint.get(n_id).minus(dbPoint.get(m_id));
        final double value = LK.product(NM) - LK.length() * NM.length() * Math.cos(dbParameter.get(param_id).getRadians());
        return TensorDouble.scalar(value);
    }

    @Override
    public void getJacobian(TensorDouble mts) {
        final Vector LK = dbPoint.get(l_id).minus(dbPoint.get(k_id));
        final Vector NM = dbPoint.get(n_id).minus(dbPoint.get(m_id));
        final Vector uLKdNM = LK.unit().product(NM.length()).product(Math.cos(dbParameter.get(param_id).getRadians()));
        final Vector uNMdLK = NM.unit().product(LK.length()).product(Math.cos(dbParameter.get(param_id).getRadians()));
        TensorDouble mt = mts;
        int j;
        //k
        j = po.get(k_id);
        mt.setVector(0, j * 2, uLKdNM.minus(NM));
        //l
        j = po.get(l_id);
        mt.setVector(0, j * 2, NM.minus(uLKdNM));
        //m
        j = po.get(m_id);
        mt.setVector(0, j * 2, uNMdLK.minus(LK));
        //n
        j = po.get(n_id);
        mt.setVector(0, j * 2, LK.minus(uNMdLK));
    }

    @Override
    public boolean isJacobianConstant() {
        return false;
    }

    @Override
    public void getHessian(TensorDouble mt, double lagrange) {

        final double L = lagrange;

        final Vector LK = dbPoint.get(l_id).minus(dbPoint.get(k_id)).unit();
        final Vector NM = dbPoint.get(n_id).minus(dbPoint.get(m_id)).unit();
        final double g = LK.product(NM) * Math.cos(dbParameter.get(param_id).getRadians());

        final TensorDouble m1GL = TensorDouble.diagonal(2, (1 - g) * L);
        final TensorDouble mG1L = TensorDouble.diagonal(2, (g - 1) * L);

        int i;
        int j;

        //k,k
        i = po.get(k_id);
        j = po.get(k_id);
        // 0

        //k,l
        i = po.get(k_id);
        j = po.get(l_id);
        //0

        //k,m
        i = po.get(k_id);
        j = po.get(m_id);
        mt.plusSubMatrix(2 * i, 2 * j, m1GL);

        //k,n
        i = po.get(k_id);
        j = po.get(n_id);
        mt.plusSubMatrix(2 * i, 2 * j, mG1L);

        //l,k
        i = po.get(l_id);
        j = po.get(k_id);
        //0

        //l,l
        i = po.get(l_id);
        j = po.get(l_id);
        // 0

        //l,m
        i = po.get(l_id);
        j = po.get(m_id);
        mt.plusSubMatrix(2 * i, 2 * j, mG1L);

        //l,n
        i = po.get(l_id);
        j = po.get(n_id);
        mt.plusSubMatrix(2 * i, 2 * j, m1GL);

        //m,k
        i = po.get(m_id);
        j = po.get(k_id);
        mt.plusSubMatrix(2 * i, 2 * j, m1GL);

        //m,l
        i = po.get(m_id);
        j = po.get(l_id);
        mt.plusSubMatrix(2 * i, 2 * j, mG1L);

        //m,m
        i = po.get(m_id);
        j = po.get(m_id);
        //0

        //m,n
        i = po.get(m_id);
        j = po.get(n_id);
        //0

        //n,k
        i = po.get(n_id);
        j = po.get(k_id);
        mt.plusSubMatrix(2 * i, 2 * j, mG1L);

        //n,l
        i = po.get(n_id);
        j = po.get(l_id);
        mt.plusSubMatrix(2 * i, 2 * j, m1GL);

        //n,m
        i = po.get(n_id);
        j = po.get(m_id);
        //0

        //n,n
        i = po.get(n_id);
        j = po.get(n_id);
        //0
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
        final Vector LK = dbPoint.get(l_id).minus(dbPoint.get(k_id));
        final Vector NM = dbPoint.get(n_id).minus(dbPoint.get(m_id));
        final TensorDouble mt = getValue();
        return mt.getQuick(0, 0) / (LK.length() * NM.length());
    }
}
