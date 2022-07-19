package com.mstruzek.msketch;

import com.mstruzek.msketch.matrix.MatrixDouble;

import static com.mstruzek.msketch.Parameter.dbParameter;
import static com.mstruzek.msketch.Point.dbPoint;

/**
 * Wiez odpowiedzialny za kat pomiedzy wektorami
 *
 * @author root
 */
public class ConstraintAngle2Lines extends Constraint {

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
     * Numer parametru przechowujacy kat w radianach
     */
    int param_id;


    /**
     * Konstruktor pomiedzy 4 punktami i paramtetrem
     * rownanie tego wiezu to (L-K)'*(N-M)-cos(param)*|L-K|*|N-M| = 0
     *
     * @param K punkt prostej
     * @param L punkt prostej
     * @param M punkt prostej
     * @param N punkt prostej
     */
    public ConstraintAngle2Lines(Integer constId, Point K, Point L, Point M, Point N, Parameter param) {
        super(constId, GeometricConstraintType.Angle2Lines, true);
        k_id = K.id;
        l_id = L.id;
        m_id = M.id;
        n_id = N.id;
        param_id = param.getId();
    }

    public String toString() {
        double norm = getNorm();
        return "Constraint-Angle2Lines" + constraintId + "*s" + size() + " = " + norm + " { K =" + dbPoint.get(k_id) + "  ,L =" + dbPoint.get(l_id) + " ,M =" + dbPoint.get(m_id) + ",N =" + dbPoint.get(n_id) + ", Parametr-" + dbParameter.get(param_id).getId() + " = " + dbParameter.get(param_id).getValue() + "} \n";
    }

    @Override
    public MatrixDouble getValue() {
        Vector LK = dbPoint.get(l_id).sub(dbPoint.get(k_id));
        Vector NM = dbPoint.get(n_id).sub(dbPoint.get(m_id));
        double value = LK.dot(NM) - LK.length() * NM.length() * Math.cos(dbParameter.get(param_id).getRadians());
        return MatrixDouble.scalar(value);
    }

    @Override
    public void getJacobian(MatrixDouble mts) {
        Vector LK = dbPoint.get(l_id).sub(dbPoint.get(k_id));
        Vector NM = dbPoint.get(n_id).sub(dbPoint.get(m_id));
        Vector uLKdNM = LK.unit().dot(NM.length()).dot(Math.cos(dbParameter.get(param_id).getRadians()));
        Vector uNMdLK = NM.unit().dot(LK.length()).dot(Math.cos(dbParameter.get(param_id).getRadians()));
        MatrixDouble mt = mts;
        int j;
        //k
        j = po.get(k_id);
        mt.setVector(0, j * 2, uLKdNM.sub(NM));
        //l
        j = po.get(l_id);
        mt.setVector(0, j * 2, NM.sub(uLKdNM));
        //m
        j = po.get(m_id);
        mt.setVector(0, j * 2, uNMdLK.sub(LK));
        //n
        j = po.get(n_id);
        mt.setVector(0, j * 2, LK.sub(uNMdLK));
    }

    @Override
    public boolean isJacobianConstant() {
        return false;
    }

    @Override
    public MatrixDouble getHessian(double lagrange) {
        MatrixDouble mt = MatrixDouble.matrix2D(dbPoint.size() * 2, dbPoint.size() * 2, 0.0);
        Vector LK = dbPoint.get(l_id).sub(dbPoint.get(k_id)).unit();
        Vector NM = dbPoint.get(n_id).sub(dbPoint.get(m_id)).unit();
        double g = LK.dot(NM) * Math.cos(dbParameter.get(param_id).getRadians());
        int i;
        int j;

        //k,k
        i = po.get(k_id);
        j = po.get(k_id);
        // 0

        //k,l
        i = po.get(k_id);
        j = po.get(l_id);
        //0

        //k,m
        i = po.get(k_id);
        j = po.get(m_id);
        mt.setSubMatrix(2 * i, 2 * j, MatrixDouble.diagonal(2, (1 - g) * lagrange));

        //k,n
        i = po.get(k_id);
        j = po.get(n_id);
        mt.setSubMatrix(2 * i, 2 * j, MatrixDouble.diagonal(2, (g - 1) * lagrange));

        //l,k
        i = po.get(l_id);
        j = po.get(k_id);
            //0

        //l,l
        i = po.get(l_id);
        j = po.get(l_id);
        // 0

        //l,m
        i = po.get(l_id);
        j = po.get(m_id);
        mt.setSubMatrix(2 * i, 2 * j, MatrixDouble.diagonal(2, (g - 1) * lagrange));

        //l,n
        i = po.get(l_id);
        j = po.get(n_id);
        mt.setSubMatrix(2 * i, 2 * j, MatrixDouble.diagonal(2, (1 - g) * lagrange));

        //m,k
        i = po.get(m_id);
        j = po.get(k_id);
        mt.setSubMatrix(2 * i, 2 * j, MatrixDouble.diagonal(2, (1 - g) * lagrange));

        //m,l
        i = po.get(m_id);
        j = po.get(l_id);
        mt.setSubMatrix(2 * i, 2 * j, MatrixDouble.diagonal(2, (g - 1) * lagrange));

        //m,m
        i = po.get(m_id);
        j = po.get(m_id);
        //0

        //m,n
        i = po.get(m_id);
        j = po.get(n_id);
        //0

        //n,k
        i = po.get(n_id);
        j = po.get(k_id);
        mt.setSubMatrix(2 * i, 2 * j, MatrixDouble.diagonal(2, (g - 1) * lagrange));

        //n,l
        i = po.get(n_id);
        j = po.get(l_id);
        mt.setSubMatrix(2 * i, 2 * j, MatrixDouble.diagonal(2, (1 - g) * lagrange));

        //n,m
        i = po.get(n_id);
        j = po.get(m_id);
        //0

        //n,n
        i = po.get(n_id);
        j = po.get(n_id);
        //0

        return mt;
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
        return param_id;
    }

    @Override
    public double getNorm() {
        Vector LK = dbPoint.get(l_id).sub(dbPoint.get(k_id));
        Vector NM = dbPoint.get(n_id).sub(dbPoint.get(m_id));
        MatrixDouble mt = getValue();
        return mt.getQuick(0, 0) / (LK.length() * NM.length());
    }
}
