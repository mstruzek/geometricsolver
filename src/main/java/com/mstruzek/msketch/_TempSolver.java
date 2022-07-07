package com.mstruzek.msketch;

import Jama.Matrix;
import com.mstruzek.msketch.matrix.BindMatrix;
import com.mstruzek.msketch.matrix.MatrixDouble;

public class _TempSolver {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		/**
		 * Rownanie A*x=b
		 * A = macierz prostokatna x = [q,lambda]'
		 * b=wektor kolumnowy prawych stron
		 * 
		 * Macierz A=
		 * [ Fq , Q' ;
		 *   Q  , 0 ];
		 *   
		 * Fq - jakobian sil
		 * Q - d(Wiezy)/dq
		 *   
		 * Wektor b = 
		 * [-(Q'*lambda - F);
		 * - Wiezy]
		 * Wiezy - wartosc wiezow
		 * lambda - mnozniki Lagrange'a  - tyle ile wiezow
		 *  
		 */
		
		/** 
		 * Nowe Zadanie :
		 * Linia, p1 zakotwiczone na s[1,1];
		 * Linia prostopadla do osi Y;
		 */
		
		Line line1 = new Line(new Vector(0.0,0.0),new Vector(4.0,0.0));
		Line line2 = new Line(new Vector(2.0,0.0),new Vector(3.0,6.0));
		Line line3 = new Line(new Vector(6.0,6.0),new Vector(10.0,5.0));
		Line line4 = new Line(new Vector(9.0,4.0),new Vector(9.0,0.0));
		ConstraintFixPoint cn1 = new ConstraintFixPoint(line1.p1,new Vector(1.0,1.0));
		ConstraintLinesPerpendicular cn2 = new ConstraintLinesPerpendicular(Constraint.nextId(),line1.p2,line1.p1,FixLine.Y.b,FixLine.Y.a);
		ConstraintConnect2Points cn3 = new ConstraintConnect2Points(Constraint.nextId(),line1.p1,line2.p1);
		ConstraintLinesPerpendicular cn4 = new ConstraintLinesPerpendicular(Constraint.nextId(),line2.p2,line2.p1,FixLine.X.b,FixLine.X.a);
		ConstraintConnect2Points cn5 = new ConstraintConnect2Points(Constraint.nextId(),line2.p2,line3.p1);
		ConstraintConnect2Points cn6 = new ConstraintConnect2Points(Constraint.nextId(),line3.p2,line4.p1);
		ConstraintConnect2Points cn7 = new ConstraintConnect2Points(Constraint.nextId(),line4.p2,line1.p2);
		//FIXME -trzeba pomyslec nad przechowywanie wiezow
		
		//teraz wyswietlamy
		
		System.out.println(GeometricPrimitive.dbPrimitives);
		System.out.println(Constraint.dbConstraint);
		System.out.println(Point.dbPoint);
		
		// Tworzymy Macierz "A" - dla tego zadania stala w czasie
		int sizeA = Point.dbPoint.size()*2 + Constraint.allLagrangeCoffSize();
		MatrixDouble A= MatrixDouble.fill(sizeA,sizeA,0.0);
		MatrixDouble Fq = GeometricPrimitive.getAllForceJacobian();
		MatrixDouble Wq = Constraint.getFullJacobian(Point.dbPoint, Parameter.dbParameter);
		//System.out.println(Wq);
		A.addSubMatrix(0, 0, Fq);
		A.addSubMatrix(Fq.getHeight(), 0, Wq);
		A.addSubMatrix(0, Fq.getWeight(), Wq.transpose()); //to musi byc jako drugie to zaoszczedzimy pamieci bez kopi

		//System.out.println(A);
		//System.out.println(A);
		//sprawdzy rank macierzy A
		//JAMA + moje
		BindMatrix mA = null;//new BindMatrix(A.m);
		
		
		//System.out.println("Rank + " + mA.rank());
		
		// Tworzymy wektor prawych stron b
		MatrixDouble b= null;
		BindMatrix mb = null;
		BindMatrix dmx = null;
		
		//A*x=b
		
		
		//MatrixDouble x = MatrixDouble.createFromArray(mx.getArray());
		//System.out.println(x.toString());//Jama nie ma swojego printowania
		//glowna pentelka
		//teraz trzeba przepisac z kazdej pozycji wartosci do punktow
		
		// x(i+1) = x(i) + dx
		// x= [q lambda]'
		
		/**
		 * ZROBIONE :trzeba pomyslec o jakims wiazaniu np punktow z macierza ,z wartosciami tak
		 * aby zmiana w macierzy uaktualniala wartosci w punktach i odwrtonie , BIND - wiazanie
		 *  
		 */
		
		BindMatrix bmX = new BindMatrix(Point.dbPoint.size()*2 + Constraint.allLagrangeCoffSize(),1);
		bmX.bind(Point.dbPoint);
		
		System.out.println(bmX);
		
		//FIXME UWAGA NIE UZYWAC TEGO PLIKU , LEPIEJ Skethc2D
		
		//2 3 iteracje i jest git
		for(int i=0;i<5;i++){
			//tworzymy macierz vector b
			b=MatrixDouble.mergeByColumn(GeometricPrimitive.getAllForce(),Constraint.getFullConstraintValues(Point.dbPoint, Parameter.dbParameter));
			b.dot(-1);
			//System.out.println(b);
			mb= new BindMatrix(b.m);
			 
			// HESSIAN
			
			A.addSubMatrix(0, 0, Fq.addC(Constraint.getFullHessian(Point.dbPoint, Parameter.dbParameter, bmX)));
			mA = new BindMatrix(A.m);
			// rozwiazjemy zadanie A*dx=b
			dmx = new BindMatrix(mA.solve(mb).getArray());
			
			bmX.plusEquals(dmx);
			bmX.copyToPoints();//uaktualniamy punkty
			
			System.out.println(Constraint.getFullConstraintValues(Point.dbPoint, Parameter.dbParameter).transposeC());
		
			Matrix nrm = new Matrix(Constraint.getFullConstraintValues(Point.dbPoint, Parameter.dbParameter).getArray());
			System.out.println(" \n"+ nrm.norm1() + "\t" + nrm.norm2() + "\t" + nrm.normF());
		}
		System.out.println(Point.dbPoint);
		//System.out.println(Constraint.dbConstraint);
		//System.out.println(cn2.getValue(Point.dbPoint, Parameter.dbParameter));
		System.out.println(GeometricPrimitive.dbPrimitives);
		System.out.println(Constraint.dbConstraint);
	}

}
/**



*/