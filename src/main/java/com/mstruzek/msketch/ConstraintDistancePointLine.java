package com.mstruzek.msketch;

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
        double norm = getNorm();
        return "Constraint-DistancePointLine" + constraintId + "*s" + size() + " = " + norm + " { K =" + dbPoint.get(k_id) + "  ,L =" + dbPoint.get(l_id) + " ,M =" + dbPoint.get(m_id) + ", Parametr-" + Parameter.dbParameter.get(param_id).getId() + " = " + Parameter.dbParameter.get(param_id).getValue() + "} \n";
    }

    @Override
    public MatrixDouble getValue() {
        Vector LK = dbPoint.get(l_id).sub(dbPoint.get(k_id));
        Vector MK = dbPoint.get(m_id).sub(dbPoint.get(k_id));
        double d = Parameter.dbParameter.get(param_id).getValue();
        double value = LK.cr(MK) * LK.cr(MK) - d * d * LK.length() * LK.length();
        return MatrixDouble.scalar(value);
    }

    @Override
    public void getJacobian(MatrixDouble mts) {
        Vector LK = dbPoint.get(l_id).sub(dbPoint.get(k_id));
        Vector MK = dbPoint.get(m_id).sub(dbPoint.get(k_id));
        Vector ML = dbPoint.get(m_id).sub(dbPoint.get(l_id));
        double d = Parameter.dbParameter.get(param_id).getValue(); /// parameter value
        double z = LK.cr(MK);
        int j;

        //k
        j = po.get(k_id);
        mts.setVector(0, 2 * j, ML.cr().dot(z * 2).add(LK.dot(2 * d * d)));

        //l
        j = po.get(l_id);
        mts.setVector(0, 2 * j, MK.cr().dot(z * -2.0).add(LK.dot(-2.0 * d * d)));

        //m
        j = po.get(m_id);
        mts.setVector(0, 2 * j, LK.cr().dot(z * 2.0));
    }

    @Override
    public double getNorm() {
        MatrixDouble mt = getValue();
        return mt.getQuick(0, 0);
    }

    @Override
    public boolean isJacobianConstant() {
        return false;
    }

    @Override
    @InstabilityBehavior(description = "equations `or Lagrange multiplier")
    public MatrixDouble getHessian(double lagrange) {
        /// macierz NxN
        MatrixDouble mt = MatrixDouble.matrix2D(dbPoint.size() * 2, dbPoint.size() * 2, 0.0);
        Vector MK = dbPoint.get(m_id).sub(dbPoint.get(k_id));
        Vector LK = dbPoint.get(l_id).sub(dbPoint.get(k_id));
        Vector ML = dbPoint.get(m_id).sub(dbPoint.get(l_id));
        double d = Parameter.dbParameter.get(param_id).getValue();
        MatrixDouble R = MatrixDouble.matrixR();
        MatrixDouble D = MatrixDouble.diagonal(2, 2 * d * d);
        MatrixDouble Dm = MatrixDouble.diagonal(2, -2 * d * d);
        double SC = MK.dot(LK.cr()); ///
        MatrixDouble mat;
        int i;
        int j;

        //k,k
        i = po.get(k_id);j = po.get(k_id);
            mat = ML.cr().cartesian(MK.cr().sub(LK.cr())).dot(2).add(Dm);
            mt.setSubMatrix(2 * i, 2 * j, mat);

        //k,l
        i = po.get(k_id);j = po.get(l_id);
            mat = R.dotC(-2.0 * SC).add(ML.cr().cartesian(MK).mult(R).dot(2.0)).add(D);
            mt.setSubMatrix(2 * i, 2 * j, mat);

        //k,m
        i = po.get(k_id);j = po.get(m_id);
            mat = R.dotC(2 * SC).add(ML.cr().cartesian(LK.cr()).dot(2.0));
            mt.setSubMatrix(2 * i, 2 * j, mat);

        //l,k
        i = po.get(l_id);j = po.get(k_id);
            mat = R.dotC(2 * SC).add(MK.cr().cartesian(ML.cr()).dot(-2.0)).add(D);
            mt.setSubMatrix(2 * i, 2 * j, mat);

        //l,l
        i = po.get(l_id);j = po.get(l_id);
            mat = MK.cr().cartesian(MK.cr()).dot(2.0).add(Dm);
            mt.setSubMatrix(2 * i, 2 * j, mat);

        //l,m
        i = po.get(l_id);j = po.get(m_id);
            mat = R.dotC(-2.0 * SC).add(MK.cr().cartesian(LK.cr()).dot(-2.0));
            mt.setSubMatrix(2 * i, 2 * j, mat);

        //m,k
        i = po.get(m_id);j = po.get(k_id);
            mat = R.dotC(-2.0 * SC).add(LK.cr().cartesian(MK.cr()).dot(-2.0));
            mt.setSubMatrix(2 * i, 2 * j, mat);

        //m,l
        i = po.get(m_id);j = po.get(l_id);
            mat = R.dotC(2.0 * SC).add(LK.cr().cartesian(MK.cr()).dot(-2.0));
            mt.setSubMatrix(2 * i, 2 * j, mat);

        //m,m
        i = po.get(m_id);j = po.get(m_id);
            mat = LK.cr().cartesian(LK.cr()).dot(2.0);
            mt.setSubMatrix(2 * i, 2 * j, mat);

        // \\\\\\\ \\\\\\ HESSIAN
        return mt.dot(lagrange);
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
    public int getParameter() {
        return param_id;
    }
}
