package com.mstruzek.msketch;

import com.mstruzek.msketch.matrix.TensorDouble;

import static com.mstruzek.msketch.ModelRegistry.dbPoint;

public class ConstraintDistancePointLine extends Constraint {

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
     * Numer parametru przechowujacy kat w radianach
     */
    int param_id;

    /**
     * Konstruktor pomiedzy 3 punktami i paramtetrem
     * rownanie tego wiezu to (L-K)x(M-K)  -d*sqrt(LK'*LK) = 0  gdzie R = Rot(PI/2) = [ 0 -1 ; 1 0]
     *
     * @param constId
     * @param K       punkt prowadzacy prostej
     * @param L       punkt prowadzacy prostej
     * @param M       punkt odlegly od prowadzacej
     */
    public ConstraintDistancePointLine(int constId, Point K, Point L, Point M, Parameter param) {
        super(constId, ConstraintType.DistancePointLine, true);
        k_id = K.id;
        l_id = L.id;
        m_id = M.id;
        param_id = param.getId();
    }

    public String toString() {
        double norm = getNorm();
        return "Constraint-DistancePointLine" + constraintId + "*s" + size() + " = " + norm + " { K =" + dbPoint.get(k_id) + "  ,L =" + dbPoint.get(l_id) + " ,M =" + dbPoint.get(m_id) + ", Parametr-" + ModelRegistry.dbParameter.get(param_id).getId() + " = " + ModelRegistry.dbParameter.get(param_id).getValue() + "} \n";
    }

    @Override
    public TensorDouble getValue() {
        final Vector LK = dbPoint.get(l_id).minus(dbPoint.get(k_id));
        final Vector MK = dbPoint.get(m_id).minus(dbPoint.get(k_id));
        final double d = ModelRegistry.dbParameter.get(param_id).getValue();
        final double value = LK.cross(MK)  -  d * LK.length();
        return TensorDouble.scalar(value);
    }

    @Override
    public void getJacobian(TensorDouble mts) {
        final Vector LK = dbPoint.get(l_id).minus(dbPoint.get(k_id));
        final Vector MK = dbPoint.get(m_id).minus(dbPoint.get(k_id));
        final double d = ModelRegistry.dbParameter.get(param_id).getValue(); /// parameter value
        TensorDouble mt = mts;
        int j;
        //k
        j = po.get(k_id);
        mt.setVector(0, 2 * j, LK.product(d / LK.length()).minus(MK.plus(LK).pivot()));

        //l
        j = po.get(l_id);
        mt.setVector(0, 2 * j, LK.product( -1.0 * d / LK.length()));

        //m
        j = po.get(m_id);
        mt.setVector(0, 2 * j, LK.pivot());
    }

    @Override
    public double getNorm() {
        TensorDouble mt = getValue();
        return mt.getQuick(0, 0);
    }

    @Override
    public boolean isJacobianConstant() {
        return false;
    }

    @Override
    @InstabilityBehavior(description = "equations `or Lagrange multiplier , update definition from jacobian equations !!")
    public void getHessian(TensorDouble mt, double lagrange) {

        if(true)
            return;

        final double L = lagrange;

        /// macierz NxN
        final Vector MK = dbPoint.get(m_id).minus(dbPoint.get(k_id));
        final Vector LK = dbPoint.get(l_id).minus(dbPoint.get(k_id));
        final Vector ML = dbPoint.get(m_id).minus(dbPoint.get(l_id));
        final double d = ModelRegistry.dbParameter.get(param_id).getValue();
        final TensorDouble R = TensorDouble.matrixR();
        final TensorDouble D = TensorDouble.diagonal(2, 2 * d * d);
        final TensorDouble Dm = TensorDouble.diagonal(2, -2 * d * d);
        final double SC = MK.product(LK.pivot()); ///
        TensorDouble mat;
        int i;
        int j;

        //k,k
        i = po.get(k_id);
        j = po.get(k_id);
        mat = ML.pivot().cartesian(MK.pivot().minus(LK.pivot())).mulitply(2).plus(Dm);
        mt.plusSubMatrix(2 * i, 2 * j, mat.mulitply(L));

        //k,l
        i = po.get(k_id);
        j = po.get(l_id);
        mat = R.multiplyC(-2.0 * SC).plus(ML.pivot().cartesian(MK).multiply(R).mulitply(2.0)).plus(D);
        mt.plusSubMatrix(2 * i, 2 * j, mat.mulitply(L));

        //k,m
        i = po.get(k_id);
        j = po.get(m_id);
        mat = R.multiplyC(2 * SC).plus(ML.pivot().cartesian(LK.pivot()).mulitply(2.0));
        mt.plusSubMatrix(2 * i, 2 * j, mat.mulitply(L));

        //l,k
        i = po.get(l_id);
        j = po.get(k_id);
        mat = R.multiplyC(2 * SC).plus(MK.pivot().cartesian(ML.pivot()).mulitply(-2.0)).plus(D);
        mt.plusSubMatrix(2 * i, 2 * j, mat.mulitply(L));

        //l,l
        i = po.get(l_id);
        j = po.get(l_id);
        mat = MK.pivot().cartesian(MK.pivot()).mulitply(2.0).plus(Dm);
        mt.plusSubMatrix(2 * i, 2 * j, mat.mulitply(L));

        //l,m
        i = po.get(l_id);
        j = po.get(m_id);
        mat = R.multiplyC(-2.0 * SC).plus(MK.pivot().cartesian(LK.pivot()).mulitply(-2.0));
        mt.plusSubMatrix(2 * i, 2 * j, mat.mulitply(L));

        //m,k
        i = po.get(m_id);
        j = po.get(k_id);
        mat = R.multiplyC(-2.0 * SC).plus(LK.pivot().cartesian(MK.pivot()).mulitply(-2.0));
        mt.plusSubMatrix(2 * i, 2 * j, mat.mulitply(L));

        //m,l
        i = po.get(m_id);
        j = po.get(l_id);
        mat = R.multiplyC(2.0 * SC).plus(LK.pivot().cartesian(MK.pivot()).mulitply(-2.0));
        mt.plusSubMatrix(2 * i, 2 * j, mat.mulitply(L));

        //m,m
        i = po.get(m_id);
        j = po.get(m_id);
        mat = LK.pivot().cartesian(LK.pivot()).mulitply(2.0);
        mt.plusSubMatrix(2 * i, 2 * j, mat.mulitply(L));
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
        return m_id;
    }

    @Override
    public int getN() {
        return -1;
    }

    @Override
    public int getParameter() {
        return param_id;
    }
}
