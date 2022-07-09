package com.mstruzek.msketch;

import com.mstruzek.msketch.matrix.MatrixDouble;

import java.util.Collections;
import java.util.Set;

/**
 * Klasa FreePoint - czyli wolny Punkt
 */
public class FreePoint extends GeometricPrimitive{

    /**
     * fix control points -punkty kontrolne zafixowane
     */
    Point a=null;
    Point b=null;
    /**
     * dynamic points - punkty dynamiczne odpowiedzialne za przemiszczanie sie lini
     */
    Point p1=null;

    /**
     * Odleglosci pomiedzy wektorami poczatkowe
     */
    double d_a_p1, d_p1_b;
    /**
     * wsp�czynnik do skalowania wzglednego dla wyznaczenia wektorow a,b
     */
    double distance=150.0;
    /**
     * Kat rotacji wzgledem osi X ,dla wyznaczenie polozenia poczatkowego dla a,b - w stopniach
     */
    double angle=Math.toRadians(40);

    public FreePoint(Vector p){
        this(GeometricPrimitive.nextId(),p);
    }

    public FreePoint(int id,Vector v00){
        super(id,GeometricPrimitiveType.FreePoint);
        if(v00 instanceof Point){
            p1=(Point) v00;
            a=new Point(p1.getId()-1,v00.sub(new Vector(distance*Math.cos(angle),distance*Math.sin(angle))));
            b=new Point(p1.getId()+1,v00.add(new Vector(distance*Math.cos(angle),distance*Math.sin(angle))));
        }else{
            // Punkty kontrolne wyznaczamy na podstawie distance,agnle : distance- to odleglosc wektora a i b od p1
            // Kolejnosc inicjalizacji ma znaczenie
            a=new Point(Point.nextId(),v00.sub(new Vector(distance*Math.cos(angle),distance*Math.sin(angle))).x,v00.sub(new Vector(distance*Math.cos(angle),distance*Math.sin(angle))).y);
            p1=new Point(Point.nextId(),v00.x,v00.y);
            b=new Point(Point.nextId(),v00.add(new Vector(distance*Math.cos(angle),distance*Math.sin(angle))).x,v00.add(new Vector(distance*Math.cos(angle),distance*Math.sin(angle))).y);
            // przelicz odleglosci
            setAssociateConstraints(null);
        }
        calculateDistance();
    }

    /**
     * Funkcja oblicza dlugosci poczatkowe pomiedzy wektorami
     */
    private void calculateDistance(){
        d_a_p1=Math.abs(p1.sub(a).length())*0.1;
        d_p1_b=Math.abs(b.sub(p1).length())*0.1;
    }

    public String toString(){
        String out=type+"*"+this.primitiveId+": {";
        out+="a="+a+",";
        out+="p1="+p1+",";
        out+="b="+b+"}\n";
        return out;
    }

    @Override
    public void recalculateControlPoints(){
        Vector va=(p1.sub(new Vector(distance*Math.cos(angle),distance*Math.sin(angle))));
        Vector vb=(p1.add(new Vector(distance*Math.cos(angle),distance*Math.sin(angle))));
        a.setLocation(va.x,va.y);
        b.setLocation(vb.x,vb.y);
        // przelicz odleglosci
        calculateDistance();

        ((ConstraintFixPoint) Constraint.dbConstraint.get(constraints[0])).setFixVector(va);
        ((ConstraintFixPoint) Constraint.dbConstraint.get(constraints[1])).setFixVector(vb);

    }

    @Override
    public MatrixDouble getJacobian(){
        MatrixDouble mt=MatrixDouble.fill(6,6,0.0);
        /**
         * k= I*k
         * [ -ks    ks     0;
         *    ks  -2ks   ks ;
         *     0    ks   -ks];

         */
        // K -mala sztywnosci
        MatrixDouble Ks=MatrixDouble.diagonal(Consts.springStiffnessLow,Consts.springStiffnessLow);
        MatrixDouble mKs=Ks.dotC(-1);


        mt.addSubMatrix(0,0,mKs).addSubMatrix(0,2,Ks);

        mt.addSubMatrix(2,0,Ks).addSubMatrix(2,2,mKs.dotC(2.0)).addSubMatrix(2,4,Ks);
        mt.addSubMatrix(4,2,Ks).addSubMatrix(4,4,mKs);
        return mt;
    }

    @Override
    public void setAssociateConstraints(Set<Integer> skipIds){
        if(skipIds==null) skipIds=Collections.emptySet();
        ConstraintFixPoint fixPointa=new ConstraintFixPoint(Constraint.nextId(skipIds),a,false);
        ConstraintFixPoint fixPointb=new ConstraintFixPoint(Constraint.nextId(skipIds),b,false);
        constraints=new int[2];
        constraints[0]=fixPointa.constraintId;
        constraints[1]=fixPointb.constraintId;
    }

    @Override
    public MatrixDouble getForce(){
        // 8 = 4*2 (4 punkty kontrolne)
        MatrixDouble force=MatrixDouble.fill(6,1,0.0);

        //F12 - sily w sprezynach
        Vector f12=p1.sub(a).unit().dot(Consts.springStiffnessLow).dot(p1.sub(a).length()-d_a_p1);
        //F23
        Vector f23=b.sub(p1).unit().dot(Consts.springStiffnessLow).dot(b.sub(p1).length()-d_p1_b);

        //F1 - silu na poszczegolne punkty
        force.addSubMatrix(0,0,new MatrixDouble(f12,true));
        //F2
        force.addSubMatrix(2,0,new MatrixDouble(f23.sub(f12),true));
        //F3
        force.addSubMatrix(4,0,new MatrixDouble(f23.dot(-1),true));

        return force;
    }

    @Override
    public int getNumOfPoints(){
        return 3; //a,p1,p2
    }

    @Override
    public int getP1(){
        return p1.id;
    }

    @Override
    public int getP2(){
        return -1;
    }

    @Override
    public int getP3(){
        return -1;
    }

    @Override
    public int[] getAllPointsId(){
        int[] out=new int[3];

        out[0]=a.getId();
        out[1]=b.getId();
        out[2]=p1.getId();
        return out;
    }

    @Override
    public int getA(){
        return a.id;
    }

    @Override
    public int getB(){
        return b.id;
    }

    @Override
    public int getC(){
        return -1;
    }

    @Override
    public int getD(){
        return -1;
    }

}
