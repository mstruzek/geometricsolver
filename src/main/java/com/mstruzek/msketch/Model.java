package com.mstruzek.msketch;

import Jama.Matrix;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.LUDecompositionQuick;
import com.mstruzek.controller.EventBus;
import com.mstruzek.controller.EventType;
import com.mstruzek.msketch.matrix.BindMatrix;
import com.mstruzek.msketch.matrix.MatrixDouble;

import java.util.ArrayList;

/**
 * Klasa w ktorej przechowujemy caly model
 * matematyczny naszego szkicownika
 *
 * @author root
 */
public final class Model{


    /**
     * To statyczna instacja a dlaczego ?
     *
     *  Dlatego iż =>
     *
     * - GeometricPrimitives    => static
     * - Parameter              => static
     * - Constraint             => static
     */


    /**
     * zmienna w ktorej przechowujemy wszystkie elementy geometryczne
     */
    public static ArrayList<GeometricPrimitive> primitivesContainer=new ArrayList<GeometricPrimitive>();
    /**
     * zmienna w ktorej przechowujemy wszystkie wiezy nalozene przez uzytkownika
     */
    public static ArrayList<Parameter> parameterContainer=new ArrayList<Parameter>();
    /**
     * zmienna w ktorej przechowujemy wszystkie wiezy nalozene przez uzytkownika
     */
    public static ArrayList<Constraint> constraintContainer=new ArrayList<Constraint>();

    MyTableModel constraintTM;

    MyTableModel primitivesTM;

    MyTableModel parametersTM;


    private Model(){

        //addLine(new Vector(0,0), new Vector(50,50));
    }

    /**
     * Rozwiaz model , przygotuj zadanie
     */
    public static void solveSystem(){

        System.out.println("*************************");
        long start=System.currentTimeMillis(); // start timing

        //pierw uakutalnic wspolrzedne punktow w kazdym GeometricPrimiticves
        // relaxForces() ;

        //teraz obliczenia

        // System.out.println("Wymiar zadania:"  + Point.dbPoint.size()*2 );
        // System.out.println("Mnozniki Lagrange'a :" + Constraint.allLagrangeSize());
        // System.out.println("Stopnie swobody : " + (Point.dbPoint.size()*2 -  Constraint.allLagrangeSize()));

        if(Point.dbPoint.size()==0) return;

        // Tworzymy Macierz "A" - dla tego zadania stala w czasie
        int sizeA=Point.dbPoint.size()*2+Constraint.allLagrangeCoffSize();
        MatrixDouble A=MatrixDouble.fill(sizeA,sizeA,0.0);
        MatrixDouble Fq=GeometricPrimitive.getAllForceJacobian();


        MatrixDouble Wq=null;//Constraint.getFullJacobian(Point.dbPoint, Parameter.dbParameter);

        BindMatrix mA=null;

        // Tworzymy wektor prawych stron b
        MatrixDouble b=null;
        BindMatrix mb=null;
        BindMatrix dmx=null;

        BindMatrix bmX=new BindMatrix(Point.dbPoint.size()*2+Constraint.allLagrangeCoffSize(),1);
        bmX.bind(Point.dbPoint);


        //FIXME - jezeli po 6 iteracjach problem ze zbieznoscia warto przeliczyc punkty kontrolne
        // czyli przeliczyc punkty i nowe Fq - macierz obliczyc

        double erri1, erri=0, delta;
        System.out.println(" Iter/Time [ms] /Norm ");
        for(int i=0;i<10;i++){
            //zerujemy macierz A

            A=MatrixDouble.fill(sizeA,sizeA,0.0);

            //Tworzymy Macierz vector b

            MatrixDouble Fr=GeometricPrimitive.getAllForce(); // Sily  - F(q)
            MatrixDouble Fi=Constraint.getFullConstraintValues(Point.dbPoint,Parameter.dbParameter); // Wiezy  - Fi(q)
            b=MatrixDouble.mergeByColumn((Fr),(Fi));
            b.dot(-1);


            //System.out.println(b);
            mb=new BindMatrix(b.m);

            // JACOBIAN
            Wq=Constraint.getFullJacobian(Point.dbPoint,Parameter.dbParameter); // Jq = d(Fi)/dq
            //HESSIAN
            MatrixDouble Hs=Constraint.getFullHessian(Point.dbPoint,Parameter.dbParameter,bmX);
            A.addSubMatrix(0,0,Fq.addC((Hs)));
            //A.addSubMatrix(0, 0, MatrixDouble.diagonal(Fq.getHeight(), 1.0)); // macierz diagonalna

            A.addSubMatrix(Fq.getHeight(),0,Wq);
            A.addSubMatrix(0,Fq.getWeight(),Wq.transpose());
            mA=new BindMatrix(A.m);
            //System.out.println("Rank + " + mA.rank());
            // rozwiazjemy zadanie A*dx=b


            //dmx = new BindMatrix(mA.solve(mb).getArray());


            /** WS*/ // JEST PRZYSPIESZENIE WYRABIA SIE w 380 ms dla 20 lini gdzie 100ms na jedna decompozycje


            DoubleMatrix2D matrix2DA=ParseToColt.toSparse(A);


            DoubleMatrix1D matrix1Db=ParseToColt.toDenseVector(b);

            //System.out.println(matrix2DA.cardinality() + " : " + (matrix1Db.size()*matrix1Db.size()));

            //System.out.println(matrix2DA);
            //System.out.println(A);
            long start2=System.currentTimeMillis(); // start timing

            /**
             *  LU Decomposition
             *
             */
            LUDecompositionQuick LU=new LUDecompositionQuick();
            LU.decompose(matrix2DA);
            LU.solve(matrix1Db);


            long stop2=System.currentTimeMillis(); // stop timing
            dmx=ParseToColt.toBindVector(matrix1Db);


            long deltat2=stop2-start2;

            //System.out.println("TimeMillis2: " + deltat2); // print execution time

            /** KONIEC  WS*/


            // jezeli chcemy symulowac na bierzaco jak sie zmieniaja wiezy to
            // wstawiamy jakis faktor dmx.times(0.3) , i<12
            bmX.plusEquals(dmx);
            bmX.copyToPoints();//uaktualniamy punkty
            //System.out.println(bmX);

            Matrix nrm=new Matrix(Constraint.getFullConstraintValues(Point.dbPoint,Parameter.dbParameter).getArray());
            System.out.println(" \n "+(i+1)+" || "+deltat2+"  ||  "+nrm.norm1()+"\n");


            //stary warunek wyjscia
            if(nrm.norm1()<0.05){
                //System.out.println("New Norm + :" + Constraint.getFullNorm(Point.dbPoint, Parameter.dbParameter));
                break;
            }

            //Matrix nforce = new Matrix(GeometricPrymitive.getAllForce().getArray());
            //System.out.println( "Force " + i + " - " + nforce.norm1());
            //minumalizujmy sily


            if(i==0){
                erri=nrm.norm1();
            }

            //liczymy zmiane bledu
            if(i>0){
                erri1=nrm.norm1();
                delta=erri1-erri;
                erri=erri1;
                if(delta>0){
                    System.out.println("CHANGES - STOP ITERATION *******\n");
                    //bmX.minusEquals(dmx);
                    //bmX.copyToPoints();//uaktualniamy punkty
                    return;
                    //relaxForces();
                    //pierw uakutalnic wspolrzedne punktow w kazdym GeometricPrimiticves
					/*	for(Integer g:GeometricPrymitive.dbPrimitives.keySet()){
						GeometricPrymitive.dbPrimitives.get(g).recalculateControlPoints();
					}*/
                }

            }
        }

        //Teraz wyswietlmy wiezy
        //System.out.println(Constraint.dbConstraint);
        //Relaksacja sprezyn -to pewnei recalculateControlPoints

        long stop=System.currentTimeMillis(); // stop timing
        long deltat=stop-start;

        System.out.println("TimeMillis: "+deltat); // print execution time

        //System.out.println(Point.dbPoint);

    }

    public static ArrayList<Parameter> parameterContainer(){
        return parameterContainer;
    }


    public static void addLine(int primitiveId,Vector v1,Vector v2){
        primitivesContainer.add(new Line(primitiveId,v1,v2));
        EventBus.send(EventType.PRIMITIVE_TABLE_FIRE_CHANGE,new Object[]{primitivesContainer.size(),primitivesContainer.size()});
    }

    public static void addLine(Vector v1,Vector v2){
        primitivesContainer.add(new Line(v1,v2));
        EventBus.send(EventType.PRIMITIVE_TABLE_FIRE_CHANGE,new Object[]{primitivesContainer.size(),primitivesContainer.size()});
    }


    public static void addCircle(int primitiveId,Vector v1,Vector v2){
        primitivesContainer.add(new Circle(primitiveId,v1,v2));
        EventBus.send(EventType.PRIMITIVE_TABLE_FIRE_CHANGE,new Object[]{primitivesContainer.size(),primitivesContainer.size()});
    }

    public static void addCircle(Vector v1,Vector v2){
        primitivesContainer.add(new Circle(v1,v2));
        EventBus.send(EventType.PRIMITIVE_TABLE_FIRE_CHANGE,new Object[]{primitivesContainer.size(),primitivesContainer.size()});
    }


    public static void addArc(int primitiveId,Vector v1,Vector v2){
        primitivesContainer.add(new Arc(primitiveId,v1,v2));
        EventBus.send(EventType.PRIMITIVE_TABLE_FIRE_CHANGE,new Object[]{primitivesContainer.size(),primitivesContainer.size()});
    }

    public static void addArc(Vector v1,Vector v2){
        primitivesContainer.add(new Arc(v1,v2));
        EventBus.send(EventType.PRIMITIVE_TABLE_FIRE_CHANGE,new Object[]{primitivesContainer.size(),primitivesContainer.size()});
    }


    public static void addPoint(int primitiveId,Vector v1){
        primitivesContainer.add(new FreePoint(primitiveId,v1));
        EventBus.send(EventType.PRIMITIVE_TABLE_FIRE_CHANGE,new Object[]{primitivesContainer.size(),primitivesContainer.size()});
    }

    public static void addPoint(Vector v1){
        primitivesContainer.add(new FreePoint(v1));
        EventBus.send(EventType.PRIMITIVE_TABLE_FIRE_CHANGE,new Object[]{primitivesContainer.size(),primitivesContainer.size()});
    }


    public static void addConstraint(GeometricConstraintType constraintType,int K,int L,int M,int N,Double paramValue){
        Point pK=null, pL=null, pM=null, pN=null;
        Parameter parameter=null;
        if(K<0){
            return;
        }
        pK=Point.dbPoint.get(K);
        pL=Point.dbPoint.get(L);
        pM=Point.dbPoint.get(M);
        pN=Point.dbPoint.get(N);

        if(paramValue != null && constraintType.isParametrized()){
            parameter=new Parameter(paramValue);
        }

        addConstraint(Constraint.nextId(),constraintType,pK,pL,pM,pN,parameter);
    }

    public static void addConstraint(int constId,GeometricConstraintType constraintType,Point K,Point L,Point M,Point N,Parameter parameter){
        switch(constraintType){
            case Connect2Points:
                constraintContainer.add(new ConstraintConnect2Points(constId,K,L));
                break;
            case FixPoint:
                constraintContainer.add(new ConstraintFixPoint(constId,K));
                break;

            case LinesPerpendicular:
                constraintContainer.add(new ConstraintLinesPerpendicular(constId,K,L,M,N));
                break;

            case LinesParallelism:
                constraintContainer.add(new ConstraintLinesParallelism(constId,K,L,M,N));
                break;
            case Tangency:
                constraintContainer.add(new ConstraintTangency(constId,K,L,M,N));
                break;

            case Distance2Points:
                constraintContainer.add(new ConstraintDistance2Points(constId,K,L,parameter));
                parameterContainer.add(parameter);
                break;
            case Angle2Lines:
                constraintContainer.add(new ConstraintAngle2Lines(constId,K,L,M,N,parameter));
                parameterContainer.add(parameter);
                break;

            case DistancePointLine:
                // FIXME - dokonczyc wiez , nie ma IMPL
                constraintContainer.add(new ConstraintDistancePointLine(constId,K,L,M,parameter));
                parameterContainer.add(parameter);
                break;

            case LinesSameLength:
                constraintContainer.add(new ConstraintLinesSameLength(constId,K,L,M,N));
                break;

            case SetVertical:
                constraintContainer.add(new ConstraintVertical(constId,K,L));
                break;
            case SetHorizontal:
                constraintContainer.add(new ConstraintHorizontal(constId,K,L));
                break;
        }
        EventBus.send(EventType.CONSTRAINT_TABLE_FIRE_CHANGE,new Object[]{constraintContainer.size(),constraintContainer.size()});
        if(parameter!=null){
            EventBus.send(EventType.PARAMETER_TABLE_FIRE_INSERT,new Object[]{parameterContainer.size(),parameterContainer.size()});
        }
    }


    public static ArrayList<GeometricPrimitive> primitivesContainer(){
        return primitivesContainer;
    }

    public static ArrayList<Constraint> constraintContainer(){
        return constraintContainer;
    }


    public static void relaxForces(){
        for(Integer g: GeometricPrimitive.dbPrimitives.keySet()){
            GeometricPrimitive.dbPrimitives.get(g).recalculateControlPoints();
        }
    }

    public static void fluctuatePoints(double coefficient){
        //FIXME dv =  [ Random Versor * coefficient] - wektor przesuniecia na kazdy prymitiw.

    }

}
