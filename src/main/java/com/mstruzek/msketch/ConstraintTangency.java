package com.mstruzek.msketch;

import com.mstruzek.msketch.matrix.TensorDouble;

import static com.mstruzek.msketch.ModelRegistry.dbPoint;

/**
 * Wiez odpowiedzialny za stycznosc Okregu do Lini
 *
 * @author root
 */
public class ConstraintTangency extends Constraint {
    /** Punkty kontrolne */

    /*** Point K-id */
    int k_id;
    /*** Point L-id */
    int l_id;
    /*** Point M-id */
    int m_id;
    /*** Point N-id */
    int n_id;

    /**
     * Konstruktor pomiedzy 4 punktami lub
     * rownanie tego wiezu to [(L-K)x(M-K)]^2 - (L-K)'*(L-K)*(N-M)'*(N-M) = 0
     *
     * @param constId
     * @param K       punkt prostej
     * @param L       punkt prostej
     * @param M       srodek okregu = p1
     * @param N       promien okregu = p2
     */
    public ConstraintTangency(int constId, Point K, Point L, Point M, Point N) {
        super(constId, ConstraintType.Tangency, true);
        k_id = K.id;
        l_id = L.id;
        m_id = M.id;
        n_id = N.id;
    }

    public String toString() {
        double norm = getNorm();
        return "Constraint-Tangency" + constraintId + "*s" + size() + " = " + norm + " { K =" + dbPoint.get(k_id) + "  ,L =" + dbPoint.get(l_id) + " ,M =" + dbPoint.get(m_id) + ",N =" + dbPoint.get(n_id) + "} \n";
    }

    @Override
    public TensorDouble getValue() {
        final Vector MK = dbPoint.get(m_id).minus(dbPoint.get(k_id));
        final Vector LK = dbPoint.get(l_id).minus(dbPoint.get(k_id));
        final Vector NM = dbPoint.get(n_id).minus(dbPoint.get(m_id));
        final double value = LK.cross(MK) * LK.cross(MK) - LK.product(LK) * NM.product(NM);
        return TensorDouble.scalar(value);
    }

    @Override
    @InstabilityBehavior(description = "rewrite equation into form  [(L-K)x(M-K)] - sqrt[(L-K)'*(L-K)]*sqrt[(N-M)'*(N-M)] = 0")
    public void getJacobian(TensorDouble mts) {
        final Vector MK = (dbPoint.get(m_id)).minus(dbPoint.get(k_id));
        final Vector LK = (dbPoint.get(l_id)).minus(dbPoint.get(k_id));
        final Vector ML = (dbPoint.get(m_id)).minus(dbPoint.get(l_id));
        final Vector NM = (dbPoint.get(n_id)).minus(dbPoint.get(m_id));
        final double nm = NM.product(NM);
        final double lk = LK.product(LK);
        final double CRS = LK.cross(MK);
        TensorDouble mt= mts;
        int j;
        //k
        j = po.get(k_id);
        mt.setVector(0, j * 2, ML.pivot().product(2.0 * CRS).plus(LK.product(2.0 * nm)));

        //l
        j = po.get(l_id);
        mt.setVector(0, j * 2, MK.pivot().product(-2.0 * CRS).plus(LK.product(-2.0 * nm)));

        //m
        j = po.get(m_id);
        mt.setVector(0, j * 2, LK.pivot().product(2.0 * CRS).plus(NM.product(2.0 * lk)));

        //n
        j = po.get(n_id);
        mt.setVector(0, j * 2, NM.product(-2.0 * lk));
    }

    @Override
    public boolean isJacobianConstant() {
        return false;
    }

    @Override
    @InstabilityBehavior(description = "equations, `or lagrange multiplier")
    public void getHessian(TensorDouble mt, double lagrange) {

        if(true) return;

        final double L = lagrange;
        /*
         * wspolczynnik lagrange ?? mat.dot(lagrange) ??
         */
        /// macierz NxN
        final Vector MK = dbPoint.get(m_id).minus(dbPoint.get(k_id));
        final Vector LK = dbPoint.get(l_id).minus(dbPoint.get(k_id));
        final Vector ML = dbPoint.get(m_id).minus(dbPoint.get(l_id));
        final Vector NM = dbPoint.get(n_id).minus(dbPoint.get(m_id));
        final TensorDouble R = TensorDouble.matrixR();

        TensorDouble mat;
        int i;
        int j;

        //k,k
        i = po.get(k_id);
        j = po.get(k_id);
        mat = ML.pivot().cartesian(ML.pivot()).mulitply(2.0).plus(TensorDouble.diagonal(2, -2.0 * NM.product(NM)));
        mt.plusSubMatrix(2 * i, 2 * j, mat.mulitply(L));

        //k,l
        i = po.get(k_id);
        j = po.get(l_id);
        mat = R.multiplyC(-2.0 * MK.product(LK.pivot())).plus(ML.pivot().cartesian(MK.pivot()).mulitply(-2.0)).plus(TensorDouble.diagonal(2, 2.0 * NM.product(NM)));
        mt.plusSubMatrix(2 * i, 2 * j, mat.mulitply(L));

        //k,m
        i = po.get(k_id);
        j = po.get(m_id);
        mat = R.multiplyC(2 * MK.product(LK.pivot())).plus(ML.pivot().cartesian(LK.pivot()).mulitply(2.0)).plus(TensorDouble.diagonal(2, -4.0 * NM.product(LK)));
        mt.plusSubMatrix(2 * i, 2 * j, mat.mulitply(L));

        //k,n
        i = po.get(k_id);
        j = po.get(n_id);
        mat = TensorDouble.diagonal(2, 4.0 * NM.product(LK));
        mt.plusSubMatrix(2 * i, 2 * j, mat.mulitply(L));

        //l,k
        i = po.get(l_id);
        j = po.get(k_id);
        mat = R.multiplyC(2.0 * MK.product(LK.pivot())).plus(MK.pivot().cartesian(MK.pivot()).mulitply(-2.0).plus(TensorDouble.diagonal(2, 2.0 * NM.product(NM))));
        mt.plusSubMatrix(2 * i, 2 * j, mat.mulitply(L));

        //l,l
        i = po.get(l_id);
        j = po.get(l_id);
        mat = MK.pivot().cartesian(MK.pivot()).mulitply(2.0).plus(TensorDouble.diagonal(2, -2.0 * NM.product(NM)));
        mt.plusSubMatrix(2 * i, 2 * j, mat.mulitply(L));

        //l,m
        i = po.get(l_id);
        j = po.get(m_id);
        mat = R.multiplyC(MK.product(LK.pivot())).plus(MK.pivot().cartesian(LK.pivot())).mulitply(-2.0).plus(TensorDouble.diagonal(2, 4.0 * NM.product(LK)));
        mt.plusSubMatrix(2 * i, 2 * j, mat.mulitply(L));

        //l,n
        i = po.get(l_id);
        j = po.get(n_id);
        mat = TensorDouble.diagonal(2, -4.0 * NM.product(LK));
        mt.plusSubMatrix(2 * i, 2 * j, mat.mulitply(L));

        //m,k
        i = po.get(m_id);
        j = po.get(k_id);
        mat = R.multiplyC(-2.0 * MK.product(LK.pivot())).plus(LK.pivot().cartesian(ML.pivot()).mulitply(2.0)).plus(TensorDouble.diagonal(2, -4.0 * NM.product(LK)));
        mt.plusSubMatrix(2 * i, 2 * j, mat.mulitply(L));

        //m,l
        i = po.get(m_id);
        j = po.get(l_id);
        mat = R.multiplyC(MK.product(LK.pivot())).mulitply(2.0).plus(LK.pivot().cartesian(MK.pivot()).mulitply(-2.0)).plus(TensorDouble.diagonal(2, 4.0 * NM.product(LK)));
        mt.plusSubMatrix(2 * i, 2 * j, mat.mulitply(L));

        //m,m
        i = po.get(m_id);
        j = po.get(m_id);
        mat = LK.pivot().cartesian(LK.pivot()).plus(TensorDouble.diagonal(2, -1.0 * LK.product(LK))).mulitply(2.0);
        mt.plusSubMatrix(2 * i, 2 * j, mat.mulitply(L));

        //m,n
        i = po.get(m_id);
        j = po.get(n_id);
        mat = TensorDouble.diagonal(2, 2.0 * LK.product(LK));
        mt.plusSubMatrix(2 * i, 2 * j, mat.mulitply(L));

        //n,k
        i = po.get(n_id);
        j = po.get(k_id);
        mat = TensorDouble.diagonal(2, 4.0 * LK.product(NM));
        mt.plusSubMatrix(2 * i, 2 * j, mat.mulitply(L));

        //n,l
        i = po.get(n_id);
        j = po.get(l_id);
        mat = TensorDouble.diagonal(2, -4.0 * LK.product(NM));
        mt.plusSubMatrix(2 * i, 2 * j, mat.mulitply(L));

        //n,m
        i = po.get(n_id);
        j = po.get(m_id);
        mat = TensorDouble.diagonal(2, 2.0 * LK.product(LK));
        mt.plusSubMatrix(2 * i, 2 * j, mat.mulitply(L));

        //n,n
        i = po.get(n_id);
        j = po.get(n_id);
        mat = TensorDouble.diagonal(2, -2.0 * LK.product(LK));
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
        return n_id;
    }

    @Override
    public int getParameter() {
        return -1;
    }

    @Override
    public double getNorm() {
        final TensorDouble mt = getValue();
        return mt.getQuick(0, 0);
    }
}
