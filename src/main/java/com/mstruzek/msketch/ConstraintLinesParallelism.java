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
    public MatrixDouble getJacobian() {
        MatrixDouble mt = MatrixDouble.matrix2D(1, dbPoint.size() * 2, 0.0);
        int j = 0;
        if ((m == null) && (n == null)) {
            for (Integer i : dbPoint.keySet()) {
                if (k_id == dbPoint.get(i).id) {
                    Vector NM = dbPoint.get(n_id).sub(dbPoint.get(m_id));
                    mt.setQuick(0, j * 2, -NM.y);
                    mt.setQuick(0, j * 2 + 1, NM.x);
                }
                if (l_id == dbPoint.get(i).id) {
                    Vector NM = dbPoint.get(n_id).sub(dbPoint.get(m_id));
                    mt.setQuick(0, j * 2, NM.y);
                    mt.setQuick(0, j * 2 + 1, -NM.x);
                }
                if (m_id == dbPoint.get(i).id) {
                    Vector LK = dbPoint.get(l_id).sub(dbPoint.get(k_id));
                    mt.setQuick(0, j * 2, LK.y);
                    mt.setQuick(0, j * 2 + 1, -LK.x);
                }
                if (n_id == dbPoint.get(i).id) {
                    Vector LK = dbPoint.get(l_id).sub(dbPoint.get(k_id));
                    mt.setQuick(0, j * 2, -LK.y);
                    mt.setQuick(0, j * 2 + 1, LK.x);
                }
                j++;
            }
        } else {
            Vector NM = n.sub(m);
            for (Integer i : dbPoint.keySet()) {
                if (k_id == dbPoint.get(i).id) {
                    mt.setQuick(0, j * 2, -NM.y);
                    mt.setQuick(0, j * 2 + 1, NM.x);
                }
                if (l_id == dbPoint.get(i).id) {
                    mt.setQuick(0, j * 2, NM.y);
                    mt.setQuick(0, j * 2 + 1, -NM.x);
                }
                j++;
            }
        }
        return mt;
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
    public MatrixDouble getHessian(double lagrange) {  /// FIXME BLAD hesianu - "unstable"
        /// macierz NxN
        MatrixDouble mt = MatrixDouble.matrix2D(dbPoint.size() * 2, dbPoint.size() * 2, 0.0);

        final MatrixDouble R = MatrixDouble.rotation(90 + 180).dot(lagrange);     /// R
        final MatrixDouble Rm = MatrixDouble.rotation(90).dot(lagrange);          /// Rm = -R
        if ((m == null) && (n == null)) {
            int i = 0;
            for (Integer vI : dbPoint.keySet()) { /// wiersz
                int j = 0;
                for (Integer vJ : dbPoint.keySet()) { /// kolumna
                    //k,m
                    if (k_id == dbPoint.get(vI).id && m_id == dbPoint.get(vJ).id) {
                        mt.addSubMatrix(2 * i, 2 * j, R);
                    }
                    //k,n
                    if (k_id == dbPoint.get(vI).id && n_id == dbPoint.get(vJ).id) {
                        mt.addSubMatrix(2 * i, 2 * j, Rm);
                    }
                    //l,m
                    if (l_id == dbPoint.get(vI).id && m_id == dbPoint.get(vJ).id) {
                        mt.addSubMatrix(2 * i, 2 * j, Rm);
                    }
                    //l,n
                    if (l_id == dbPoint.get(vI).id && n_id == dbPoint.get(vJ).id) {
                        mt.addSubMatrix(2 * i, 2 * j, R);
                    }
                    //m,k
                    if (m_id == dbPoint.get(vI).id && k_id == dbPoint.get(vJ).id) {
                        mt.addSubMatrix(2 * i, 2 * j, Rm);
                    }
                    //m,l
                    if (m_id == dbPoint.get(vI).id && l_id == dbPoint.get(vJ).id) {
                        mt.addSubMatrix(2 * i, 2 * j, R);
                    }
                    //n,k
                    if (n_id == dbPoint.get(vI).id && k_id == dbPoint.get(vJ).id) {
                        mt.addSubMatrix(2 * i, 2 * j, R);
                    }
                    //n,l
                    if (n_id == dbPoint.get(vI).id && l_id == dbPoint.get(vJ).id) {
                        mt.addSubMatrix(2 * i, 2 * j, Rm);
                    }
                    j++;
                }
                i++;
            }

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
