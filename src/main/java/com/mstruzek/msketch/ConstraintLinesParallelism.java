package com.mstruzek.msketch;

import com.mstruzek.msketch.matrix.MatrixDouble;

import static com.mstruzek.msketch.Point.dbPoint;

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
        super(constId, GeometricConstraintType.LinesParallelism, true);
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
        double norm = getNorm();
        if (m == null && n == null)
            return "Constraint-LinesParallelism" + constraintId + "*s" + size() + " = " + norm + " { K =" + dbPoint.get(k_id) + "  ,L =" + dbPoint.get(l_id) + " ,M =" + dbPoint.get(m_id) + ",N =" + dbPoint.get(n_id) + "} \n";
        else {
            return "Constraint-LinesParallelism" + constraintId + "*s" + size() + " = " + norm + " { K =" + dbPoint.get(k_id) + "  ,L =" + dbPoint.get(l_id) + " ,vecM =" + m + ",vecN =" + n + "} \n";
        }
    }

    @Override
    public MatrixDouble getValue() {
        Vector LK = dbPoint.get(l_id).sub(dbPoint.get(k_id));
        if ((m == null) && (n == null)) {
            double value = LK.cr(dbPoint.get(n_id).sub(dbPoint.get(m_id)));
            return MatrixDouble.scalar(value);
        } else {
            double value = LK.cr(n.sub(m));
            return MatrixDouble.scalar(value);
        }
    }

    @Override
    public void getJacobian(MatrixDouble mts) {
        MatrixDouble mt = mts;
        int i;
        if ((m == null) && (n == null)) {
            Vector NM = dbPoint.get(n_id).sub(dbPoint.get(m_id));
            Vector LK = dbPoint.get(l_id).sub(dbPoint.get(k_id));
            //k
            i = po.get(k_id);
            mt.setVector(0, i * 2, NM.  cr());
            //l
            i = po.get(l_id);
            mt.setVector(0, i * 2, NM.cr().dot(-1.0));
            //m
            i = po.get(m_id);
            mt.setVector(0, i * 2, LK.cr().dot(-1.0));
            //n
            i = po.get(n_id);
            mt.setVector(0, i * 2, LK.cr());

        } else {
            Vector NM = n.sub(m);
            //k
            i = po.get(k_id);
            mt.setVector(0, i * 2, NM.cr());
            //l
            i = po.get(l_id);
            mt.setVector(0, i * 2, NM.cr().dot(-1.0));
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
    public MatrixDouble getHessian(double lagrange) {
        /// macierz NxN
        MatrixDouble mt = MatrixDouble.matrix2D(dbPoint.size() * 2, dbPoint.size() * 2, 0.0);
        final MatrixDouble R = MatrixDouble.rotation(90 + 180).dot(lagrange);     /// R
        final MatrixDouble Rm = MatrixDouble.rotation(90).dot(lagrange);          /// Rm = -R
        int i;
        int j;
        if ((m == null) && (n == null)) {

            //k,m
            i = po.get(k_id);
            j = po.get(m_id);
            mt.addSubMatrix(2 * i, 2 * j, R);

            //k,n
            i = po.get(k_id);
            j = po.get(n_id);
            mt.addSubMatrix(2 * i, 2 * j, Rm);

            //l,m
            i = po.get(l_id);
            j = po.get(m_id);
            mt.addSubMatrix(2 * i, 2 * j, Rm);

            //l,n
            i = po.get(l_id);
            j = po.get(n_id);
            mt.addSubMatrix(2 * i, 2 * j, R);

            //m,k
            i = po.get(m_id);
            j = po.get(k_id);
            mt.addSubMatrix(2 * i, 2 * j, Rm);

            //m,l
            i = po.get(m_id);
            j = po.get(l_id);
            mt.addSubMatrix(2 * i, 2 * j, R);

            //n,k
            i = po.get(n_id);
            j = po.get(k_id);
            mt.addSubMatrix(2 * i, 2 * j, R);

            //n,l
            i = po.get(n_id);
            j = po.get(l_id);
            mt.addSubMatrix(2 * i, 2 * j, Rm);

            return mt;
        } else {
            // HESSIAN ZERO //m,n - vectory
            return null;
        }
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
        Vector LK = dbPoint.get(k_id).sub(dbPoint.get(l_id));
        MatrixDouble mt = getValue();
        if ((m == null) && (n == null)) {
            return mt.getQuick(0, 0) / LK.length() / dbPoint.get(m_id).sub(dbPoint.get(n_id)).length();
        } else {
            return mt.getQuick(0, 0) / LK.length() / (m.sub(n)).length();
        }
    }
}
