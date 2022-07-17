package com.mstruzek.msketch;

import com.mstruzek.msketch.matrix.MatrixDouble;

import static com.mstruzek.msketch.Point.dbPoint;

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
        super(constId, GeometricConstraintType.Tangency, true);
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
    public MatrixDouble getValue() {
        Vector MK = dbPoint.get(m_id).sub(dbPoint.get(k_id));
        Vector LK = dbPoint.get(l_id).sub(dbPoint.get(k_id));
        Vector NM = dbPoint.get(n_id).sub(dbPoint.get(m_id));
        var value = LK.cr(MK) * LK.cr(MK) - LK.dot(LK) * NM.dot(NM);
        return MatrixDouble.scalar(value);
    }

    @Override
    public MatrixDouble getJacobian() {
        MatrixDouble mt = MatrixDouble.matrix2D(1, dbPoint.size() * 2, 0.0);
        Vector MK = (dbPoint.get(m_id)).sub(dbPoint.get(k_id));
        Vector LK = (dbPoint.get(l_id)).sub(dbPoint.get(k_id));
        Vector ML = (dbPoint.get(m_id)).sub(dbPoint.get(l_id));
        Vector NM = (dbPoint.get(n_id)).sub(dbPoint.get(m_id));
        double nm = NM.dot(NM);
        double lk = LK.dot(LK);
        double CRS = LK.cr(MK);
        int j = 0;
        for (Integer i : dbPoint.keySet()) {
            // K
            if (k_id == dbPoint.get(i).id) {
                mt.setVector(0, j * 2, ML.cr().dot(2.0 * CRS).add(LK.dot(2.0 * nm)));
            }
            // L
            if (l_id == dbPoint.get(i).id) {
                mt.setVector(0, j * 2, MK.cr().dot(-2.0 * CRS).add(LK.dot(-2.0 * nm)));
            }
            // M
            if (m_id == dbPoint.get(i).id) {
                mt.setVector(0, j * 2, LK.cr().dot(2.0 * CRS).add(NM.dot(2.0 * lk)));
            }
            // N
            if (n_id == dbPoint.get(i).id) {
                mt.setVector(0, j * 2, NM.dot(-2.0 * lk));
            }
            j++;
        }
        // System.out.println(mt.toString(dbPoint.keySet().toArray(new Integer[0])));
        return mt;
    }

    @Override
    public boolean isJacobianConstant() {
        return false;
    }

    @Override
    @InstabilityBehavior(description = "equations, `or lagrange multiplier")
    public MatrixDouble getHessian(double lagrange) {
        /*
         * wspolczynnik lagrange ?? mat.dot(lagrange) ??
         */
        /// macierz NxN
        MatrixDouble mt = MatrixDouble.matrix2D(dbPoint.size() * 2, dbPoint.size() * 2, 0.0);
        Vector MK = dbPoint.get(m_id).sub(dbPoint.get(k_id));
        Vector LK = dbPoint.get(l_id).sub(dbPoint.get(k_id));
        Vector ML = dbPoint.get(m_id).sub(dbPoint.get(l_id));
        Vector NM = dbPoint.get(n_id).sub(dbPoint.get(m_id));
        MatrixDouble R = MatrixDouble.matrixR();
        int i = 0;
        /// same punkty
        for (Integer qI : dbPoint.keySet()) { //wiersz
            int j = 0;
            for (Integer qJ : dbPoint.keySet()) { //kolumna
                //k,k
                if (k_id == dbPoint.get(qI).id && k_id == dbPoint.get(qJ).id) {
                    var mat = ML.cr().cartesian(ML.cr()).dot(2.0).add(MatrixDouble.diagonal(2, -2.0 * NM.dot(NM)));
                    mt.setSubMatrix(2 * i, 2 * j, mat);
                }
                //k,l
                if (k_id == dbPoint.get(qI).id && l_id == dbPoint.get(qJ).id) {
                    var mat = R.dotC(-2.0 * MK.dot(LK.cr())).add(ML.cr().cartesian(MK.cr()).dot(-2.0)).add(MatrixDouble.diagonal(2, 2.0 * NM.dot(NM)));
                    mt.setSubMatrix(2 * i, 2 * j, mat);
                }
                //k,m
                if (k_id == dbPoint.get(qI).id && m_id == dbPoint.get(qJ).id) {
                    var mat = R.dotC(2 * MK.dot(LK.cr())).add(ML.cr().cartesian(LK.cr()).dot(2.0)).add(MatrixDouble.diagonal(2, -4.0 * NM.dot(LK)));
                    mt.setSubMatrix(2 * i, 2 * j, mat);
                }
                //k,n
                if (k_id == dbPoint.get(qI).id && n_id == dbPoint.get(qJ).id) {
                    var mat = MatrixDouble.diagonal(2, 4.0 * NM.dot(LK));
                    mt.setSubMatrix(2 * i, 2 * j, mat);
                }
                //l,k
                if (l_id == dbPoint.get(qI).id && k_id == dbPoint.get(qJ).id) {
                    var mat = R.dotC(2.0 * MK.dot(LK.cr())).add(MK.cr().cartesian(MK.cr()).dot(-2.0).add(MatrixDouble.diagonal(2, 2.0 * NM.dot(NM))));
                    mt.setSubMatrix(2 * i, 2 * j, mat);
                }
                //l,l
                if (l_id == dbPoint.get(qI).id && l_id == dbPoint.get(qJ).id) {
                    var mat = MK.cr().cartesian(MK.cr()).dot(2.0).add(MatrixDouble.diagonal(2, -2.0 * NM.dot(NM)));
                    mt.setSubMatrix(2 * i, 2 * j, mat);
                }
                //l,m
                if (l_id == dbPoint.get(qI).id && m_id == dbPoint.get(qJ).id) {
                    var mat = R.dotC(MK.dot(LK.cr())).add(MK.cr().cartesian(LK.cr())).dot(-2.0).add(MatrixDouble.diagonal(2, 4.0 * NM.dot(LK)));
                    mt.setSubMatrix(2 * i, 2 * j, mat);
                }
                //l,n
                if (l_id == dbPoint.get(qI).id && n_id == dbPoint.get(qJ).id) {
                    var mat = MatrixDouble.diagonal(2, -4.0 * NM.dot(LK));
                    mt.setSubMatrix(2 * i, 2 * j, mat);
                }
                //m,k
                if (m_id == dbPoint.get(qI).id && k_id == dbPoint.get(qJ).id) {
                    var mat = R.dotC(-2.0 * MK.dot(LK.cr())).add(LK.cr().cartesian(ML.cr()).dot(2.0)).add(MatrixDouble.diagonal(2, -4.0 * NM.dot(LK)));
                    mt.setSubMatrix(2 * i, 2 * j, mat);
                }
                //m,l
                if (m_id == dbPoint.get(qI).id && l_id == dbPoint.get(qJ).id) {
                    var mat = R.dotC(MK.dot(LK.cr())).dot(2.0).add(LK.cr().cartesian(MK.cr()).dot(-2.0)).add(MatrixDouble.diagonal(2, 4.0 * NM.dot(LK)));
                    mt.setSubMatrix(2 * i, 2 * j, mat);
                }
                //m,m
                if (m_id == dbPoint.get(qI).id && m_id == dbPoint.get(qJ).id) {
                    var mat = LK.cr().cartesian(LK.cr()).add(MatrixDouble.diagonal(2, -1.0 * LK.dot(LK))).dot(2.0);
                    mt.setSubMatrix(2 * i, 2 * j, mat);
                }
                //m,n
                if (m_id == dbPoint.get(qI).id && n_id == dbPoint.get(qJ).id) {
                    var mat = MatrixDouble.diagonal(2, 2.0 * LK.dot(LK));
                    mt.setSubMatrix(2 * i, 2 * j, mat);
                }
                //n,k
                if (n_id == dbPoint.get(qI).id && k_id == dbPoint.get(qJ).id) {
                    var mat = MatrixDouble.diagonal(2, 4.0 * LK.dot(NM));
                    mt.setSubMatrix(2 * i, 2 * j, mat);
                }
                //n,l
                if (n_id == dbPoint.get(qI).id && l_id == dbPoint.get(qJ).id) {
                    var mat = MatrixDouble.diagonal(2, -4.0 * LK.dot(NM));
                    mt.setSubMatrix(2 * i, 2 * j, mat);
                }
                //n,m
                if (n_id == dbPoint.get(qI).id && m_id == dbPoint.get(qJ).id) {
                    var mat = MatrixDouble.diagonal(2, 2.0 * LK.dot(LK));
                    mt.setSubMatrix(2 * i, 2 * j, mat);
                }
                //n,n
                if (n_id == dbPoint.get(qI).id && n_id == dbPoint.get(qJ).id) {
                    var mat = MatrixDouble.diagonal(2, -2.0 * LK.dot(LK));
                    mt.setSubMatrix(2 * i, 2 * j, mat);
                }
                j++;
            }
            i++;
        }
        /// TODO aktualnie = 1.0   <==      langrange = 2.106151810818924E-8
//        out.println("langrange = " + lagrange);
//        out.println(mt.toString(dbPoint.keySet().toArray(new Integer[0])));
//        return mt;
//        return mt.dot(lagrange);      /// ????
        return null;
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
        MatrixDouble mt = getValue();
        return mt.getQuick(0, 0);
    }
}
