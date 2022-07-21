package com.mstruzek.msketch;

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

    /*** Punkty kontrolne */

    /*** Point K-id */
    int k_id;
    /*** Point L-id */
    int l_id;
    /*** Point M-id */
    int m_id;
    /*** Vector M - gdy wiez pomiedzy fixline */
    Vector m = null;
    /*** Point N-id */
    int n_id;
    /*** Vector N - gdy wiez pomiedzy fixline */
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
        double norm = getNorm();
        if (m == null && n == null)
            return "Constraint-LinesPerpendicular" + constraintId + "*s" + size() + " = " + norm + " { K =" + dbPoint.get(k_id) + "  ,L =" + dbPoint.get(l_id) + " ,M =" + dbPoint.get(m_id) + ",N =" + dbPoint.get(n_id) + "} \n";
        else {
            return "Constraint-LinesPerpendicular" + constraintId + "*s" + size() + " = " + norm + " { K =" + dbPoint.get(k_id) + "  ,L =" + dbPoint.get(l_id) + " ,vecM =" + m + ",vecN =" + n + "} \n";
        }

    }

    @Override
    public void getJacobian(MatrixDouble mts) {
        MatrixDouble mt = mts;
        int j = 0;
        if ((m == null) && (n == null)) {
            /// K
            j = po.get(k_id);
            mt.setVector(0, j * 2, dbPoint.get(m_id).Vector().minus(dbPoint.get(n_id)));
            /// L
            j = po.get(l_id);
            mt.setVector(0, j * 2, dbPoint.get(m_id).Vector().minus(dbPoint.get(n_id)).product(-1.0));
            /// M
            j = po.get(m_id);
            mt.setVector(0, j * 2, dbPoint.get(k_id).Vector().minus(dbPoint.get(l_id)));
            /// N
            j = po.get(n_id);
            mt.setVector(0, j * 2, dbPoint.get(k_id).Vector().minus(dbPoint.get(l_id)).product(-1.0));

        } else {
            /// K
            j = po.get(k_id);
            mt.setVector(0, j * 2, m.minus(n));
            /// L
            j = po.get(l_id);
            mt.setVector(0, j * 2, m.minus(n).product(-1.0));
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
    public MatrixDouble getValue() {
        if ((m == null) && (n == null)) {
            double value = (dbPoint.get(k_id).minus(dbPoint.get(l_id))).product(dbPoint.get(m_id).minus(dbPoint.get(n_id)));
            return MatrixDouble.scalar(value);
        } else {
            double value = (dbPoint.get(k_id).minus(dbPoint.get(l_id))).product(m.minus(n));
            return MatrixDouble.scalar(value);
        }
    }

    @Override
    public MatrixDouble getHessian(double lagrange) {
        /// macierz NxN
        MatrixDouble mt = MatrixDouble.matrix2D(dbPoint.size() * 2, dbPoint.size() * 2, 0.0);
        MatrixDouble I = MatrixDouble.identity(2, 1.0 * lagrange);
        MatrixDouble Im = MatrixDouble.identity(2, 1.0 * lagrange);
        int i;
        int j;
        if ((m == null) && (n == null)) {
            //wstawiamy I,-I w odpowiednie miejsca
            /// K,M
            i = po.get(k_id);
            j = po.get(m_id);
            mt.setSubMatrix(2 * i, 2 * j, I);

            /// K,N
            i = po.get(k_id);
            j = po.get(n_id);
            mt.setSubMatrix(2 * i, 2 * j, Im);

            /// L,M
            i = po.get(l_id);
            j = po.get(m_id);
            mt.setSubMatrix(2 * i, 2 * j, Im);

            /// L,N
            i = po.get(l_id);
            j = po.get(n_id);
            mt.setSubMatrix(2 * i, 2 * j, I);

            /// M,K
            i = po.get(m_id);
            j = po.get(k_id);
            mt.setSubMatrix(2 * i, 2 * j, I);

            /// M,L
            i = po.get(m_id);
            j = po.get(l_id);
            mt.setSubMatrix(2 * i, 2 * j, Im);

            /// N,K
            i = po.get(n_id);
            j = po.get(k_id);
            mt.setSubMatrix(2 * i, 2 * j, Im);

            /// N,L
            i = po.get(n_id);
            j = po.get(l_id);
            mt.setSubMatrix(2 * i, 2 * j, I);

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
    public int getParameter() {
        return -1;
    }

    @Override
    public double getNorm() {
        Vector vKL = dbPoint.get(k_id).minus(dbPoint.get(l_id));
        MatrixDouble mt = getValue();
        if ((m == null) && (n == null)) {
            return mt.getQuick(0, 0) / vKL.length() / dbPoint.get(m_id).minus(dbPoint.get(n_id)).length();
        } else {
            return mt.getQuick(0, 0) / vKL.length() / (m.minus(n)).length();
        }
    }

}
