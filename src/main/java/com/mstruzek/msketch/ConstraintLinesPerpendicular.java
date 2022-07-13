package com.mstruzek.msketch;

import Jama.Matrix;
import com.mstruzek.msketch.matrix.MatrixDouble;

import static com.mstruzek.msketch.Point.dbPoint;

/**
 * Klasa reprezentujaca wiez prosopadlosci pomiedzy dwoma wektorami
 * skladajacymi sie z 4 punktow
 * lub 2 punktow i jednego FixLine
 *
 * @author root
 */
public class ConstraintLinesPerpendicular extends Constraint {

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
     * rownanie tego wiezu to (K-L)'*(M-N) = 0
     * iloczyn skalarny
     *
     * @param constId
     * @param K
     * @param L
     * @param M
     * @param N
     */
    public ConstraintLinesPerpendicular(int constId, Point K, Point L, Vector M, Vector N) {
        super(constId, GeometricConstraintType.LinesPerpendicular, true);
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
        MatrixDouble out = getValue();
        double norm = Matrix.constructWithCopy(out.getArray()).norm1();
        if (m == null && n == null)
            return "Constraint-LinesPerpendicular" + constraintId + "*s" + size() + " = " + norm + " { K =" + dbPoint.get(k_id) + "  ,L =" + dbPoint.get(l_id) + " ,M =" + dbPoint.get(m_id) + ",N =" + dbPoint.get(n_id) + "} \n";
        else {
            return "Constraint-LinesPerpendicular" + constraintId + "*s" + size() + " = " + norm + " { K =" + dbPoint.get(k_id) + "  ,L =" + dbPoint.get(l_id) + " ,vecM =" + m + ",vecN =" + n + "} \n";
        }

    }

    @Override
    public MatrixDouble getJacobian() {
        /// macierz 1xN
        MatrixDouble mt = MatrixDouble.fill(1, dbPoint.size() * 2, 0.0);
        int j = 0;
        if ((m == null) && (n == null)) {
            for (Integer i : dbPoint.keySet()) {
                if (k_id == dbPoint.get(i).id) {
                    mt.setVectorT(0, j * 2, dbPoint.get(m_id).Vector().sub(dbPoint.get(n_id)));
                }
                if (l_id == dbPoint.get(i).id) {
                    mt.setVectorT(0, j * 2, dbPoint.get(m_id).Vector().sub(dbPoint.get(n_id)).dot(-1.0));
                }
                if (m_id == dbPoint.get(i).id) {
                    mt.setVectorT(0, j * 2, dbPoint.get(k_id).Vector().sub(dbPoint.get(l_id)));
                }
                if (n_id == dbPoint.get(i).id) {
                    mt.setVectorT(0, j * 2, dbPoint.get(k_id).Vector().sub(dbPoint.get(l_id)).dot(-1.0));
                }
                j++;
            }
        } else {
            for (Integer i : dbPoint.keySet()) {
                if (k_id == dbPoint.get(i).id) {
                    mt.setVectorT(0, j * 2, m.sub(n));
                }
                if (l_id == dbPoint.get(i).id) {
                    mt.setVectorT(0, j * 2, m.sub(n).dot(-1.0));
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
    public MatrixDouble getValue() {
        MatrixDouble mt = new MatrixDouble(1, 1);
        if ((m == null) && (n == null)) {
            mt.set(0, 0, (dbPoint.get(k_id).sub(dbPoint.get(l_id))).dot(dbPoint.get(m_id).sub(dbPoint.get(n_id))));
        } else {
            mt.set(0, 0, (dbPoint.get(k_id).sub(dbPoint.get(l_id))).dot(m.sub(n)));
        }
        return mt;
    }

    @Override
    public MatrixDouble getHessian(double alfa) {
        /// macierz NxN
        MatrixDouble mt = MatrixDouble.fill(dbPoint.size() * 2, dbPoint.size() * 2, 0.0);
        if ((m == null) && (n == null)) {
            MatrixDouble I = MatrixDouble.identity(2).dot(alfa);
            MatrixDouble Im = MatrixDouble.identity(2).dot(alfa);
            //same punkty
            int i = 0;
            for (Integer vI : dbPoint.keySet()) { //wiersz
                int j = 0;
                for (Integer vJ : dbPoint.keySet()) { //kolumna
                    //wstawiamy I,-I w odpowiednie miejsca
                    //k,m
                    if (k_id == dbPoint.get(vI).id && m_id == dbPoint.get(vJ).id) {
                        mt.setSubMatrix(2 * i, 2 * j, I);
                    }
                    //k,n
                    if (k_id == dbPoint.get(vI).id && n_id == dbPoint.get(vJ).id) {
                        mt.setSubMatrix(2 * i, 2 * j, Im);
                    }
                    //l,m
                    if (l_id == dbPoint.get(vI).id && m_id == dbPoint.get(vJ).id) {
                        mt.setSubMatrix(2 * i, 2 * j, Im);
                    }
                    //l,n
                    if (l_id == dbPoint.get(vI).id && n_id == dbPoint.get(vJ).id) {
                        mt.setSubMatrix(2 * i, 2 * j, I);
                    }
                    //m,k
                    if (m_id == dbPoint.get(vI).id && k_id == dbPoint.get(vJ).id) {
                        mt.setSubMatrix(2 * i, 2 * j, I);
                    }
                    //m,l
                    if (m_id == dbPoint.get(vI).id && l_id == dbPoint.get(vJ).id) {
                        mt.setSubMatrix(2 * i, 2 * j, Im);
                    }
                    //n,k
                    if (n_id == dbPoint.get(vI).id && k_id == dbPoint.get(vJ).id) {
                        mt.setSubMatrix(2 * i, 2 * j, Im);
                    }
                    //n,l
                    if (n_id == dbPoint.get(vI).id && l_id == dbPoint.get(vJ).id) {
                        mt.setSubMatrix(2 * i, 2 * j, I);
                    }
                    j++;
                }
                i++;
            }
            return mt;
        } else {
            /// m,n - vectory
            /// HESSIAN ZERO
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
    public int getParametr() {
        return -1;
    }

    @Override
    public double getNorm() {
        Vector vKL = dbPoint.get(k_id).sub(dbPoint.get(l_id));
        MatrixDouble mt = getValue();
        if ((m == null) && (n == null)) {
            return mt.get(0, 0) / vKL.length() / dbPoint.get(m_id).sub(dbPoint.get(n_id)).length();
        } else {
            return mt.get(0, 0) / vKL.length() / (m.sub(n)).length();
        }
    }

}
