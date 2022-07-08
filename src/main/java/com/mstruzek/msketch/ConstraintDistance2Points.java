package com.mstruzek.msketch;

import java.util.TreeMap;

import Jama.Matrix;
import com.mstruzek.msketch.matrix.MatrixDouble;

/**
 * Klasa reprezentujaca wiez odleglosci pomiedzy 2 punktami
 *
 * @author root
 */
public class ConstraintDistance2Points extends Constraint{

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
     * Numer parametru
     */
    int param_id;

    /**
     * Konstruktor pomiedzy 2 punktami
     * rownanie tego wiezu to sqrt[(K-L)'*(K-L)] - d = 0
     *
     * @param constId
     * @param K
     * @param L
     */
    public ConstraintDistance2Points(int constId,Point K,Point L,Parameter param){
        super(constId, GeometricConstraintType.Distance2Points);
        k_id=K.id;
        l_id=L.id;
        param_id=param.getId();
    }

    public String toString(){
        MatrixDouble out=getValue(Point.dbPoint,Parameter.dbParameter);
        double norm=Matrix.constructWithCopy(out.getArray()).norm1();
        return "Constraint-Distance2Points"+constraintId+"*s"+size()+" = "+norm+" { K ="+Point.dbPoint.get(k_id)+"  ,L ="+Point.dbPoint.get(l_id)+" , Parametr-"+Parameter.dbParameter.get(param_id).getId()+" = "+Parameter.dbParameter.get(param_id).getValue()+" } \n";

    }

    @Override
    public MatrixDouble getJacobian(TreeMap<Integer,Point> dbPoints,TreeMap<Integer,Parameter> dbParameter){
        //macierz 2 wierszowa
        MatrixDouble out=MatrixDouble.fill(1,dbPoints.size()*2,0.0);
        //zerujemy cala macierz + wstawiamy na odpowiednie miejsce Jacobian wiezu
        int j=0;
        Vector vLK=((Vector) dbPoints.get(l_id)).sub((Vector) dbPoints.get(k_id)).unit();
        for(Integer i: dbPoints.keySet()){

            //a tu wstawiamy macierz dla tego wiezu
            if(k_id==dbPoints.get(i).id){
                out.m[0][j*2]=-vLK.x;
                out.m[0][j*2+1]=-vLK.y;
            }
            if(l_id==dbPoints.get(i).id){
                out.m[0][j*2]=vLK.x;
                out.m[0][j*2+1]=vLK.y;
            }
            j++;
        }

        return out;
    }

    @Override
    public boolean isJacobianConstant(){
        return true;

    }

    @Override
    public MatrixDouble getValue(TreeMap<Integer,Point> dbPoints,TreeMap<Integer,Parameter> dbParameter){

        Double vLK=((Vector) dbPoints.get(l_id)).sub((Vector) dbPoints.get(k_id)).length();

        MatrixDouble mt=new MatrixDouble(1,1);
        mt.m[0][0]=vLK-dbParameter.get(param_id).getRadians();
        return mt;
    }

    @Override
    public MatrixDouble getHessian(TreeMap<Integer,Point> dbPoints,TreeMap<Integer,Parameter> dbParameter){

        //macierz NxN
        MatrixDouble out=MatrixDouble.fill(dbPoints.size()*2,dbPoints.size()*2,0.0);

        return out;
    }

    @Override
    public boolean isHessianConstant(){
        return true;
    }

    @Override
    public int getK(){
        return k_id;
    }

    @Override
    public int getL(){
        return l_id;
    }

    @Override
    public int getM(){
        return -1;
    }

    @Override
    public int getN(){
        return -1;
    }

    @Override
    public int getParametr(){
        return param_id;
    }

    /**
     * @param args
     */
    public static void main(String[] args){

        Point p1=new Point(Point.nextId(),0.0,0.1);
        Point p2=new Point(Point.nextId(),1.0,0.2);
        Parameter par=new Parameter(1.0);
        ConstraintDistance2Points con=new ConstraintDistance2Points(Constraint.nextId(),p1,p2,new Parameter(0.8));
        ConstraintDistance2Points con2=new ConstraintDistance2Points(Constraint.nextId(),p1,p2,par);
        System.out.println(con);


    }

    @Override
    public double getNorm(TreeMap<Integer,Point> dbPoints,TreeMap<Integer,Parameter> dbParameter){

        Double vLK=((Vector) dbPoints.get(l_id)).sub((Vector) dbPoints.get(k_id)).length();

        return (vLK-dbParameter.get(param_id).getRadians());
    }

}
