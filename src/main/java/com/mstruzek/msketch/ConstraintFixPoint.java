package com.mstruzek.msketch;

import Jama.Matrix;
import com.mstruzek.msketch.matrix.MatrixDouble;

import static com.mstruzek.msketch.Point.dbPoint;

/**
 * Klasa reprezentuje wiez typu FIX : "PointK - VectorK0 =0";
 *
 * @author root
 */
public class ConstraintFixPoint extends Constraint {

    /**
     * Point K-id
     */
    int k_id;
    /**
     * Vector L
     */
    Vector k0_vec = null;

    /**
     * Konstruktor wiezu
     *
     * @param constId
     * @param K       - Point zafiksowany bedzie w aktualnym miejscu
     */
    public ConstraintFixPoint(int constId, Point K) {
        this(constId, K, true);
    }

    /**
     * Konstruktor wiezu
     *
     * @param K - zrzutowany Point na Vector
     * @param L - Vector w ktorym bedzie zafiksowany K
     */
    public ConstraintFixPoint(Point K, Vector L) {
        super(nextId(), GeometricConstraintType.FixPoint, true);
        //pobierz id
        k_id = ((Point) K).id;
        k0_vec = L;
        dbConstraint.put(constraintId, this);
    }

    public ConstraintFixPoint(Integer constId,Point K,boolean storage){
        super(constId, GeometricConstraintType.FixPoint, storage);
        this.k_id = K.id;
        this.k0_vec = new Vector(K.x, K.y);
    }

    /**
     * Funkcja ustawia nowy wektor zafiksowania
     *
     * @param vc
     */
    public void setFixVector(Vector vc) {
        k0_vec = vc;
    }

    public String toString() {
        MatrixDouble out = getValue();
        double norm = Matrix.constructWithCopy(out.getArray()).norm1();

        return "Constraint-FixPoint" + constraintId + "*s" + size() + " = " + norm + " { K =" + dbPoint.get(k_id) + "  , K0 = " + k0_vec + " } \n";
    }

    @Override
    public MatrixDouble getJacobian() {
        //macierz 2 wierszowa
        MatrixDouble out = MatrixDouble.fill(2, dbPoint.size() * 2, 0.0);
        //zerujemy cala macierz + wstawiamy na odpowiednie miejsce Jacobian wiezu
        int j = 0;
        for(Integer i : dbPoint.keySet()) {
            if(k_id == dbPoint.get(i).id) {
                //macierz jednostkowa
                out.m[0][j * 2] = 1.0;
                out.m[0][j * 2 + 1] = 0.0;
                out.m[1][j * 2] = 0.0;
                out.m[1][j * 2 + 1] = 1.0;
            }
            j++;
        }
        return out;
    }

    @Override
    public boolean isJacobianConstant() {
        return true;
    }

    @Override
    public MatrixDouble getValue() {
        return new MatrixDouble(((Vector) dbPoint.get(k_id)).sub(k0_vec), true);
    }

    @Override
    public MatrixDouble getHessian() {
        return null;
    }

    @Override
    public boolean isHessianConstant() {
        return false;
    }


    @Override
    public int getK() {
        return k_id;
    }

    @Override
    public int getL() {
        return -1;
    }

    @Override
    public int getM() {
        return -1;
    }

    @Override
    public int getN() {
        return -1;
    }

    @Override
    public int getParametr() {
        return -1;
    }

    public static void main(String[] args) {

        Point pn = new Point(Point.nextId(), new Vector(3.0, 4.0).x, new Vector(3.0, 4.0).y);
        Point pn3 = new Point(Point.nextId(), new Vector(3.0, 4.0).x, new Vector(3.0, 4.0).y);
        Point pn2 = new Point(Point.nextId(), new Vector(4.0, 5.0).x, new Vector(4.0, 5.0).y);

        ConstraintConnect2Points conectPoint = new ConstraintConnect2Points(Constraint.nextId(), pn, pn2);


        ConstraintFixPoint fixPoint2 = new ConstraintFixPoint(Constraint.nextId(), pn3);
        System.out.println(Constraint.dbConstraint);
        //jakobian z wiezow
        //double[][] tab = fixPoint2.getJacobian2D(Point.dbPoint, Parameter.dbParameter);
        //MatrixDouble tab2 = conectPoint.getJacobian(Point.dbPoint, Parameter.dbParameter);

        //tak sie zabieramy za wielkosc wektora
        System.out.println(fixPoint2.getJacobian());
        System.out.println(conectPoint.getJacobian());

    }

    @Override
    public double getNorm() {

        MatrixDouble md = getValue();
        double val = Math.sqrt(md.m[0][0] * md.m[0][0] + md.m[1][0] * md.m[1][0]);
        return val;
    }
}
