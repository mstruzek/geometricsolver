package com.mstruzek.msketch;

import Jama.Matrix;
import com.mstruzek.msketch.matrix.MatrixDouble;

import static com.mstruzek.msketch.Point.dbPoint;

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
     * rownanie tego wiezu to [(R(L-K))'*(M-K)]^2 - d*d*(L-K)'*(L-K) = 0  gdzie R = Rot(PI/2) = [ 0 -1 ; 1 0]
     *
     * @param constId
     * @param K       punkt prowadzacy prostej
     * @param L       punkt prowadzacy prostej
     * @param M       punkt odlegly od prowadzacej
     */
    public ConstraintDistancePointLine(int constId, Point K, Point L, Point M, Parameter param) {
        super(constId, GeometricConstraintType.DistancePointLine, true);
        k_id = K.id;
        l_id = L.id;
        m_id = M.id;
        param_id = param.getId();
    }

    public String toString() {
        MatrixDouble mt = getValue();
        double norm = Matrix.constructWithCopy(mt.getArray()).norm1();
        return "Constraint-DistancePointLine" + constraintId + "*s" + size() + " = " + norm + " { K =" + dbPoint.get(k_id) + "  ,L =" + dbPoint.get(l_id) + " ,M =" + dbPoint.get(m_id) + ", Parametr-" + Parameter.dbParameter.get(param_id).getId() + " = " + Parameter.dbParameter.get(param_id).getValue() + "} \n";
    }

    @Override
    public MatrixDouble getValue() {
        MatrixDouble mt = new MatrixDouble(1, 1);
        Vector LK = dbPoint.get(l_id).sub(dbPoint.get(k_id));
        Vector MK = dbPoint.get(m_id).sub(dbPoint.get(k_id));
        double d = Parameter.dbParameter.get(param_id).getValue();

        mt.set(0, 0, LK.cross(MK) * LK.cross(MK) - d * d * LK.length() * LK.length());

        return mt;
    }

    @Override
    public MatrixDouble getJacobian() {
        MatrixDouble mt = new MatrixDouble(1, dbPoint.size() * 2);
        Vector LK = dbPoint.get(l_id).sub(dbPoint.get(k_id));
        Vector MK = dbPoint.get(m_id).sub(dbPoint.get(k_id));
        Vector ML = dbPoint.get(m_id).sub(dbPoint.get(l_id));
        double d = Parameter.dbParameter.get(param_id).getValue(); /// parameter value
        double z = LK.cross(MK);
        int j = 0;
        for (Integer id : dbPoint.keySet()) {
            /// ################################################################## k  /// Explicitly Repeated Accessor
            if (k_id == dbPoint.get(id).id) {
                mt.setVectorT(0, 2 * j, ML.cross().dot(z * 2).add(LK.dot(2 * d * d)));
            }
            /// ################################################################## l
            if (l_id == dbPoint.get(id).id) {
                mt.setVectorT(0, 2 * j, MK.cross().dot(z * -2.0).add(LK.dot(-2.0 * d * d)));
            }
            /// ################################################################## m
            if (m_id == dbPoint.get(id).id) {
                mt.setVectorT(0, 2 * j, LK.cross().dot(z * 2.0));
            }
            j++;
        }
        return mt;
    }

    @Override
    public double getNorm() {
        MatrixDouble mt = getValue();
        return mt.get(0, 0);
    }

    @Override
    public boolean isJacobianConstant() {
        return false;
    }

    @Override
    public MatrixDouble getHessian(double lagrange) {
        /// macierz NxN
        MatrixDouble mt = MatrixDouble.fill(dbPoint.size() * 2, dbPoint.size() * 2, 0.0);
        Vector MK = dbPoint.get(m_id).sub(dbPoint.get(k_id));
        Vector LK = dbPoint.get(l_id).sub(dbPoint.get(k_id));
        Vector ML = dbPoint.get(m_id).sub(dbPoint.get(l_id));
        double d = Parameter.dbParameter.get(param_id).getValue();
        MatrixDouble R = MatrixDouble.matR();
        MatrixDouble D = MatrixDouble.diagonal(2, 2 * d * d);
        MatrixDouble Dm = MatrixDouble.diagonal(2, -2 * d * d);
        double SC = MK.dot(LK.cross()); ///
        int i = 0;
        for (Integer qI : dbPoint.keySet()) { // wiersz
            int j = 0;
            for (Integer qJ : dbPoint.keySet()) { // kolumna
                /// # # # FI - k
                //k,k
                if (k_id == dbPoint.get(qI).id && k_id == dbPoint.get(qJ).id) {
                    var mat = ML.cross().cartesian(MK.cross().sub(LK.cross())).dot(2).add(Dm);
                    mt.setSubMatrix(2 * i, 2 * j, mat);
                }
                //k,l
                if (k_id == dbPoint.get(qI).id && l_id == dbPoint.get(qJ).id) {
                    var mat = R.dotC(-2.0 * SC).add(ML.cross().cartesian(MK).mult(R).dot(2.0)).add(D);
                    mt.setSubMatrix(2 * i, 2 * j, mat);
                }
                //k,m
                if (k_id == dbPoint.get(qI).id && m_id == dbPoint.get(qJ).id) {
                    var mat = R.dotC(2 * SC).add(ML.cross().cartesian(LK.cross()).dot(2.0));
                    mt.setSubMatrix(2 * i, 2 * j, mat);
                }
                /// # # # FI - l
                //l,k
                if (l_id == dbPoint.get(qI).id && k_id == dbPoint.get(qJ).id) {
                    var mat = R.dotC(2 * SC).add(MK.cross().cartesian(ML.cross()).dot(-2.0)).add(D);
                    mt.setSubMatrix(2 * i, 2 * j, mat);
                }
                //l,l
                if (l_id == dbPoint.get(qI).id && l_id == dbPoint.get(qJ).id) {
                    var mat = MK.cross().cartesian(MK.cross()).dot(2.0).add(Dm);
                    mt.setSubMatrix(2 * i, 2 * j, mat);
                }
                //l,m
                if (l_id == dbPoint.get(qI).id && m_id == dbPoint.get(qJ).id) {
                    var mat = R.dotC(-2.0 * SC).add(MK.cross().cartesian(LK.cross()).dot(-2.0));
                    mt.setSubMatrix(2 * i, 2 * j, mat);
                }
                /// # # # FI - m
                //m,k
                if (m_id == dbPoint.get(qI).id && k_id == dbPoint.get(qJ).id) {
                    var mat = R.dotC(-2.0 * SC).add( LK.cross().cartesian( MK.cross()).dot(-2.0));
                    mt.setSubMatrix(2 * i, 2 * j, mat);
                }
                //m,l
                if (m_id == dbPoint.get(qI).id && l_id == dbPoint.get(qJ).id) {
                    var mat = R.dotC(2.0 * SC).add( LK.cross().cartesian( MK.cross()).dot(-2.0));
                    mt.setSubMatrix(2 * i, 2 * j, mat);
                }
                //m,m
                if (m_id == dbPoint.get(qI).id && m_id == dbPoint.get(qJ).id) {
                    var mat = LK.cross().cartesian( LK.cross()).dot(2.0);
                    mt.setSubMatrix(2 * i, 2 * j, mat);
                }
                j++;
            }
            i++;
        }
        return mt;
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
    public int getParametr() {
        return param_id;
    }
}
