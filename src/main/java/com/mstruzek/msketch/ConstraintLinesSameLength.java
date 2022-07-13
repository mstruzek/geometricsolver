package com.mstruzek.msketch;

import Jama.Matrix;
import com.mstruzek.msketch.matrix.MatrixDouble;

import static com.mstruzek.msketch.Point.dbPoint;

/**
 * Klasa reprezentujaca wiez rownej dlugosci pomiedzy
 * dwoma wektorami (4 punkty)
 *
 * @author root
 */
public class ConstraintLinesSameLength extends Constraint {

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
     * Konstruktor pomiedzy 4 punktami lub
     * 2 punktami i FixLine(czyli 2 wektory)
     * rownanie tego wiezu to sqrt[(K-L)'*(K-L)] - sqrt[(M-N)'*(M-N)] = 0
     * iloczyn skalarny
     *
     * @param constId
     * @param K
     * @param L
     * @param M
     * @param N
     */
    public ConstraintLinesSameLength(int constId, Point K, Point L, Point M, Point N) {
        this(constId, K, L, M, N, true);
    }

    public ConstraintLinesSameLength(int constId, Point K, Point L, Point M, Point N, boolean storage) {
        super(constId, GeometricConstraintType.LinesSameLength, storage);
        k_id = K.id;
        l_id = L.id;
        m_id = M.id;
        n_id = N.id;
    }

    public String toString() {
        MatrixDouble out = getValue();
        double norm = Matrix.constructWithCopy(out.getArray()).norm1();
        return "Constraint-LinesSameLength" + constraintId + "*s" + size() + " = " + norm + " { K =" + dbPoint.get(k_id) + "  ,L =" + dbPoint.get(l_id) + " ,M =" + dbPoint.get(m_id) + ",N =" + dbPoint.get(n_id) + "} \n";
    }

    @Override
    public MatrixDouble getJacobian() {
        MatrixDouble mt = MatrixDouble.fill(1, dbPoint.size() * 2, 0.0);
        Vector vLK = dbPoint.get(l_id).sub(dbPoint.get(k_id)).unit();
        Vector vNM = dbPoint.get(n_id).sub(dbPoint.get(m_id)).unit();
        int j = 0;
        for (Integer i : dbPoint.keySet()) {
            if (k_id == dbPoint.get(i).id) {
                mt.setVectorT(0, j * 2 , vLK.dot(-1.0));
            }
            if (l_id == dbPoint.get(i).id) {
                mt.setVectorT(0, j * 2 , vLK);
            }
            if (m_id == dbPoint.get(i).id) {
                mt.setVectorT(0, j * 2 , vNM);
            }
            if (n_id == dbPoint.get(i).id) {
                mt.setVectorT(0, j * 2 , vNM.dot(-1.0));
            }
            j++;
        }
        return mt;
    }

    @Override
    public boolean isJacobianConstant() {
        return true;

    }

    @Override
    public MatrixDouble getValue() {
        MatrixDouble mt = new MatrixDouble(1, 1);
        Double vLK = dbPoint.get(l_id).sub(dbPoint.get(k_id)).length();
        Double vNM = dbPoint.get(n_id).sub(dbPoint.get(m_id)).length();
        mt.set(0, 0 , vLK - vNM);
        return mt;
    }

    @Override
    public MatrixDouble getHessian(double alfa) {
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
    public int getParametr() {
        return -1;
    }

    @Override
    public double getNorm() {
        MatrixDouble md = getValue();
        return md.get(0, 0);
    }

}
