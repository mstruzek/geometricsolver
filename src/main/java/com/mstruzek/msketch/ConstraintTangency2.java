package com.mstruzek.msketch;

import Jama.Matrix;
import com.mstruzek.msketch.matrix.MatrixDouble;

import static com.mstruzek.msketch.Point.dbPoint;

/**
 * Wiez odpowiedzialny za stycznosc Okregu do Lini
 *
 * @author root
 */
public class ConstraintTangency2 extends Constraint {

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
     * macierz rotacji o 90 stopni
     */
    static MatrixDouble R = MatrixDouble.getRotation2x2(90 + 180);
    static MatrixDouble mR = MatrixDouble.getRotation2x2(90); //mR=-R

    /**
     * Konstruktor pomiedzy 4 punktami lub
     * rownanie tego wiezu to |(L-K)x(M-K)| - |L-K|*|N-M|= 0
     *
     * @param constId
     * @param K       punkt prostej
     * @param L       punkt prostej
     * @param M       srodek okregu = p1
     * @param N       promien okregu = p2
     */
    public ConstraintTangency2(int constId, Point K, Point L, Point M, Point N) {
        super(constId, GeometricConstraintType.Tangency, true);
        k_id = K.id;
        l_id = L.id;
        m_id = M.id;
        n_id = N.id;
    }

    public String toString() {
        MatrixDouble mt = getValue();
        double norm = Matrix.constructWithCopy(mt.getArray()).norm1();
        return "Constraint-Tangency" + constraintId + "*s" + size() + " = " + norm + " { K =" + dbPoint.get(k_id) + "  ,L =" + dbPoint.get(l_id) + " ,M =" + dbPoint.get(m_id) + ",N =" + dbPoint.get(n_id) + "} \n";
    }

    @Override
    public MatrixDouble getValue() {
        Vector MK = dbPoint.get(m_id).sub(dbPoint.get(k_id));
        Vector LK = dbPoint.get(l_id).sub(dbPoint.get(k_id));
        Vector NM = dbPoint.get(n_id).sub(dbPoint.get(m_id));
        MatrixDouble mt = new MatrixDouble(1, 1);
        mt.set(0, 0, Math.sqrt(LK.cr(MK) * LK.cr(MK)) - LK.length() * NM.length());
        return mt;
    }

    @Override
    public MatrixDouble getJacobian() {
        MatrixDouble mt = MatrixDouble.fill(1, dbPoint.size() * 2, 0.0);
        Vector MK = (dbPoint.get(m_id)).sub(dbPoint.get(k_id));
        Vector LK = (dbPoint.get(l_id)).sub(dbPoint.get(k_id));
        Vector LM = (dbPoint.get(l_id)).sub(dbPoint.get(m_id));
        Vector NM = (dbPoint.get(n_id)).sub(dbPoint.get(m_id));
        double g = LK.cr(MK);
        double zn = Math.signum(g);
        int j = 0;
        for (Integer i : dbPoint.keySet()) {
            //a tu wstawiamy macierz dla tego wiezu - IOT
            if (k_id == dbPoint.get(i).id) {
                mt.set(0, j * 2     ,  zn * LM.y + NM.length() * LK.unit().x);
                mt.set(0, j * 2 + 1 , -zn * LM.x + NM.length() * LK.unit().y);
            }
            if (l_id == dbPoint.get(i).id) {
                mt.set(0, j * 2     ,  zn * MK.y - NM.length() * LK.unit().x);
                mt.set(0, j * 2 + 1 , -zn * MK.x - NM.length() * LK.unit().y);
            }
            //a tu wstawiamy macierz dla tego wiezu
            if (m_id == dbPoint.get(i).id) {
                mt.set(0, j * 2     , -zn * LK.y + LK.length() * NM.unit().x);
                mt.set(0, j * 2 + 1 ,  zn * LK.x + LK.length() * NM.unit().y);
            }
            if (n_id == dbPoint.get(i).id) {
                mt.set(0, j * 2     , -LK.length() * NM.unit().x);
                mt.set(0, j * 2 + 1 , -LK.length() * NM.unit().y);
            }
            j++;
        }
        return mt;
    }

    @Override
    public boolean isJacobianConstant() {
        return false;
    }

    @Override
    public MatrixDouble getHessian(double lagrange) {
        //macierz NxN
        MatrixDouble mt = MatrixDouble.fill(dbPoint.size() * 2, dbPoint.size() * 2, 0.0);
        Vector MK = dbPoint.get(m_id).sub(dbPoint.get(k_id));
        Vector LK = dbPoint.get(l_id).sub(dbPoint.get(k_id));
        Vector LM = dbPoint.get(l_id).sub(dbPoint.get(m_id));
        Vector NM = dbPoint.get(n_id).sub(dbPoint.get(m_id));
        double g = LK.cr(MK);
        double zn = Math.signum(g);
        double k = NM.unit().dot(LK.unit());
        int i = 0;
        //same punkty
        for (Integer qI : dbPoint.keySet()) { //wiersz         /// FIXME -- outer loop
            int j = 0;
            for (Integer qJ : dbPoint.keySet()) { //kolumna
                //k,k
                if (k_id == dbPoint.get(qI).id && k_id == dbPoint.get(qJ).id) {
                    // 0
                }
                //k,l
                if (k_id == dbPoint.get(qI).id && l_id == dbPoint.get(qJ).id) {
                    mt.setSubMatrix(2 * i, 2 * j, R.dotC(zn).dot(lagrange));
                }
                //k,m
                if (k_id == dbPoint.get(qI).id && m_id == dbPoint.get(qJ).id) {
                    mt.setSubMatrix(2 * i, 2 * j, mR.dotC(zn).add(MatrixDouble.diag(2, -k)).dot(lagrange));
                }
                //k,n
                if (k_id == dbPoint.get(qI).id && n_id == dbPoint.get(qJ).id) {
                    mt.setSubMatrix(2 * i, 2 * j, MatrixDouble.diag(2, k).dot(lagrange));
                }
                //l,k
                if (l_id == dbPoint.get(qI).id && k_id == dbPoint.get(qJ).id) {
                    mt.setSubMatrix(2 * i, 2 * j, mR.dotC(zn).dot(lagrange));
                }
                //l,l
                if (l_id == dbPoint.get(qI).id && l_id == dbPoint.get(qJ).id) {
                    // 0
                }
                //l,m
                if (l_id == dbPoint.get(qI).id && m_id == dbPoint.get(qJ).id) {
                    mt.setSubMatrix(2 * i, 2 * j, R.dotC(zn).add(MatrixDouble.diag(2, k)).dot(lagrange));
                }
                //l,n
                if (l_id == dbPoint.get(qI).id && n_id == dbPoint.get(qJ).id) {
                    mt.setSubMatrix(2 * i, 2 * j, MatrixDouble.diag(2, -k).dot(lagrange));
                }
                //m,k
                if (m_id == dbPoint.get(qI).id && k_id == dbPoint.get(qJ).id) {
                    mt.setSubMatrix(2 * i, 2 * j, R.dotC(zn).add(MatrixDouble.diag(2, -k)).dot(lagrange));
                }
                //m,l
                if (m_id == dbPoint.get(qI).id && l_id == dbPoint.get(qJ).id) {
                    mt.setSubMatrix(2 * i, 2 * j, mR.dotC(zn).add(MatrixDouble.diag(2, k)).dot(lagrange));
                }
                //m,m
                if (m_id == dbPoint.get(qI).id && m_id == dbPoint.get(qJ).id) {
                    //0
                }
                //m,n
                if (m_id == dbPoint.get(qI).id && n_id == dbPoint.get(qJ).id) {
                    // 0
                }
                //n,k
                if (n_id == dbPoint.get(qI).id && k_id == dbPoint.get(qJ).id) {
                    mt.setSubMatrix(2 * i, 2 * j, MatrixDouble.diag(2, k).dot(lagrange));
                }
                //n,l
                if (n_id == dbPoint.get(qI).id && l_id == dbPoint.get(qJ).id) {
                    mt.setSubMatrix(2 * i, 2 * j, MatrixDouble.diag(2, -k).dot(lagrange));
                }
                //n,m
                if (n_id == dbPoint.get(qI).id && m_id == dbPoint.get(qJ).id) {
                    // 0
                }
                //n,n
                if (n_id == dbPoint.get(qI).id && n_id == dbPoint.get(qJ).id) {
                    // 0
                }
                j++;
            }
            i++;
        }
//        return mt;
        return null;
//        return MatrixDouble.fill(dbPoint.size() * 2, dbPoint.size() * 2, 0.0);
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
        return mt.get(0,0);
    }
}
