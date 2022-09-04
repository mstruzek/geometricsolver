package com.mstruzek.msketch;

import com.mstruzek.msketch.matrix.TensorDouble;

import static com.mstruzek.msketch.ModelRegistry.dbPoint;

/**
 * Klasa reprezentujaca wiez rownoleglo�ci pomiedzy
 * liniami(vectorami)
 *
 * @author root
 */
public class ConstraintLinesParallelism extends Constraint {

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
     * Vector M - gdy wiez pomiedzy fixline
     */
    Vector m = null;
    /**
     * Point N-id
     */
    int n_id;
    /**
     * Vector N - gdy wiez pomiedzy fixline
     */
    Vector n = null;

    /**
     * Konstruktor pomiedzy 4 punktami lub
     * 2 punktami i FixLine(czyli 2 wektory)
     * rownanie tego wiezu to (L-K)x(N-M) = 0
     * iloczyn wektorowy
     * FIXME - zastanowic sie czy nie zrobic abs((L-K)x(N-M)) =0 moze bedzie szybciej zbiegal
     *
     * @param constId
     * @param K
     * @param L
     * @param M
     * @param N
     */
    public ConstraintLinesParallelism(int constId, Point K, Point L, Vector M, Vector N) {
        super(constId, ConstraintType.LinesParallelism, true);
        k_id = K.id;
        l_id = L.id;
        if ((M instanceof Point) && (N instanceof Point)) {
            m_id = ((Point) M).id;
            n_id = ((Point) N).id;
        } else {
            m = M;
            n = N;
        }
    }

    public String toString() {
        final double norm = getNorm();
        if (m == null && n == null)
            return "Constraint-LinesParallelism" + constraintId + "*s" + size() + " = " + norm + " { K =" + dbPoint.get(k_id) + "  ,L =" + dbPoint.get(l_id) + " ,M =" + dbPoint.get(m_id) + ",N =" + dbPoint.get(n_id) + "} \n";
        else {
            return "Constraint-LinesParallelism" + constraintId + "*s" + size() + " = " + norm + " { K =" + dbPoint.get(k_id) + "  ,L =" + dbPoint.get(l_id) + " ,vecM =" + m + ",vecN =" + n + "} \n";
        }
    }

    @Override
    public TensorDouble getValue() {
        final Vector LK = dbPoint.get(l_id).minus(dbPoint.get(k_id));
        if ((m == null) && (n == null)) {
            double value = LK.cross(dbPoint.get(n_id).minus(dbPoint.get(m_id)));
            return TensorDouble.scalar(value);
        } else { // not-used
            double value = LK.cross(n.minus(m));
            return TensorDouble.scalar(value);
        }
    }

    @Override
    public void getJacobian(TensorDouble mts) {
        TensorDouble mt = mts;
        int i;
        if ((m == null) && (n == null)) {
            final Vector NM = dbPoint.get(n_id).minus(dbPoint.get(m_id));
            final Vector LK = dbPoint.get(l_id).minus(dbPoint.get(k_id));
            //k
            i = po.get(k_id);
            mt.setVector(0, i * 2, NM.pivot());
            //l
            i = po.get(l_id);
            mt.setVector(0, i * 2, NM.pivot().product(-1.0));
            //m
            i = po.get(m_id);
            mt.setVector(0, i * 2, LK.pivot().product(-1.0));
            //n
            i = po.get(n_id);
            mt.setVector(0, i * 2, LK.pivot());

        } else {
            final Vector NM = n.minus(m);
            //k
            i = po.get(k_id);
            mt.setVector(0, i * 2, NM.pivot());
            //l
            i = po.get(l_id);
            mt.setVector(0, i * 2, NM.pivot().product(-1.0));
        }
    }


    @Override
    public boolean isJacobianConstant() {
        if ((m == null) && (n == null)) {
            return false;
        } else {
            //jezeli m,n vectory to constatnt i wtedy Hessian =0
            return true;
        }
    }

    @Override
    public void getHessian(TensorDouble mt, double lagrange) {

        final double L = lagrange;

        final TensorDouble R = TensorDouble.rotation(90 + 180).mulitply(L);     /// R
        final TensorDouble Rm = TensorDouble.rotation(90).mulitply(L);          /// Rm = -R
        int i;
        int j;
        if ((m == null) && (n == null)) {

            //k,m
            i = po.get(k_id);
            j = po.get(m_id);
            mt.plusSubMatrix(2 * i, 2 * j, R);

            //k,n
            i = po.get(k_id);
            j = po.get(n_id);
            mt.plusSubMatrix(2 * i, 2 * j, Rm);

            //l,m
            i = po.get(l_id);
            j = po.get(m_id);
            mt.plusSubMatrix(2 * i, 2 * j, Rm);

            //l,n
            i = po.get(l_id);
            j = po.get(n_id);
            mt.plusSubMatrix(2 * i, 2 * j, R);

            //m,k
            i = po.get(m_id);
            j = po.get(k_id);
            mt.plusSubMatrix(2 * i, 2 * j, Rm);

            //m,l
            i = po.get(m_id);
            j = po.get(l_id);
            mt.plusSubMatrix(2 * i, 2 * j, R);

            //n,k
            i = po.get(n_id);
            j = po.get(k_id);
            mt.plusSubMatrix(2 * i, 2 * j, R);

            //n,l
            i = po.get(n_id);
            j = po.get(l_id);
            mt.plusSubMatrix(2 * i, 2 * j, Rm);

        }
        // HESSIAN ZERO //m,n - vectory
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
        final Vector LK = dbPoint.get(k_id).minus(dbPoint.get(l_id));
        final TensorDouble mt = getValue();
        if ((m == null) && (n == null)) {
            final Vector MN = dbPoint.get(m_id).minus(dbPoint.get(n_id));
            return mt.getQuick(0, 0) / LK.length() / MN.length();
        } else {
            return mt.getQuick(0, 0) / LK.length() / (m.minus(n)).length();
        }
    }
}
