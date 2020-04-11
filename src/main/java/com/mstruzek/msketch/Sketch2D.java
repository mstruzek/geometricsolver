package com.mstruzek.msketch;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Line2D;
import java.util.TreeMap;

import javax.swing.JPanel;

import com.mstruzek.msketch.matrix.BindMatrix;
import com.mstruzek.msketch.matrix.MatrixDouble;
import Jama.Matrix;


public class Sketch2D extends JPanel {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	static TreeMap<Integer,GeometricPrymitive> dbPrimitives = null;
	//AffineTransform transform;
	public Sketch2D(int width,int height){
		super();
		setSize(width,height);
		setLayout(null);
		
		//tu obliczenia
		
		/** 
		 * Zadanie : Prostokat + Okrag styczny do kazdej z lini
		 * prawy dolnych rog zafiksowany  + line1 i line2 prostopadle
		 * do osci Y i X 
		 */
		
		//prostokat + okrag
		Line line1 = new Line(new Vector(0.0,0.0),new Vector(40.0,0.0));
		Line line2 = new Line(new Vector(20.0,10.0),new Vector(30.0,60.0));
		Line line3 = new Line(new Vector(40.0,60.0),new Vector(100.0,50.0));
		Line line4 = new Line(new Vector(90.0,40.0),new Vector(90.0,0.0));
		Circle cl= new Circle(new Vector(30.0,30.0),new Vector(40.0,40.0));
		Circle c2= new Circle(new Vector(1.0,1.0),new Vector(20.0,20.0));
		//Circle c3= new Circle(new Vector(-10.0,0.0),new Vector(20.0,20.0));

		//trojkat
		Line line5 = new Line(new Vector(0.0,0.0),new Vector(90.0,0.0));
		Line line6 = new Line(new Vector(90.0,0.0),new Vector(50.0,50.0));
		Line line7 = new Line(new Vector(.0,25.0),new Vector(0.0,0.0));
		
		ConstraintFixPoint cn1 = new ConstraintFixPoint(line1.p1,new Vector(20.0,10.0));
		//ConstraintFixPoint cl1 = new ConstraintFixPoint(cl.p1,new Vector(30.0,31.0));
			//ConstraintFixPoint cn10 = new ConstraintFixPoint(line2.p2,new Vector(.8,7.0));
			//ConstraintFixPoint cn12 = new ConstraintFixPoint(line1.p1,new Vector(1.0,1.0));//gdy wiez zostanie powielony to macierz A bedzie miala mniejszy rank
		
		ConstraintConect2Points cn3 = new ConstraintConect2Points(line1.p1,line2.p1);
		ConstraintConect2Points cn5 = new ConstraintConect2Points(line2.p2,line3.p1);
		ConstraintConect2Points cn6 = new ConstraintConect2Points(line3.p2,line4.p1);
		ConstraintConect2Points cn7 = new ConstraintConect2Points(line4.p2,line1.p2);
	
		//trojakt
		
		ConstraintConect2Points tcn1 = new ConstraintConect2Points(line5.p2,line6.p1);
		ConstraintConect2Points tcn2 = new ConstraintConect2Points(line6.p2,line7.p1);
		ConstraintConect2Points tcn3 = new ConstraintConect2Points(line7.p2,line5.p1);
		ConstraintFixPoint tn1 = new ConstraintFixPoint(c2.p1,new Vector(30.8,7.07));
		/*// STARE ROZWIAZANIE NA PROSOPADTLOSC
			ConstraintsLinesPerpendicular cn2 = new ConstraintsLinesPerpendicular(line1.p2,line1.p1,FixLine.Y.b,FixLine.Y.a);
			//ConstraintsLinesPerpendicular cn4 = new ConstraintsLinesPerpendicular(line2.p2,line2.p1,FixLine.X.b,FixLine.X.a);
		ConstraintsLinesPerpendicular cn12 = new ConstraintsLinesPerpendicular(line4.p2,line4.p1,line1.p2,line1.p1); //4 z 1
		ConstraintsLinesPerpendicular cn4 = new ConstraintsLinesPerpendicular(line2.p2,line2.p1,line1.p2,line1.p1); //2 z 1
		ConstraintsLinesPerpendicular cn8 = new ConstraintsLinesPerpendicular(line3.p2,line3.p1,line2.p1,line2.p2); // 3 z 2
			//ConstraintsLinesPerpendicular cn19 = new ConstraintsLinesPerpendicular(line4.p2,line4.p1,FixLine.X.b,FixLine.X.a);
		//ConstraintsLinesPerpendicular cn9 = new ConstraintsLinesPerpendicular(line4.p2,line4.p1,line3.p2,line3.p1); //4 z 3
	*
	*/
		//ConstraintsLinesPerpendicular cn23 = new ConstraintsLinesPerpendicular(line2.p2,line2.p1,line1.p2,line1.p1);
		ConstraintLinesPerpendicular cn2 = new ConstraintLinesPerpendicular(line1.p2,line1.p1,FixLine.Y.b,FixLine.Y.a);
		ConstraintLinesPerpendicular cn2x = new ConstraintLinesPerpendicular(line2.p2,line2.p1,FixLine.X.b,FixLine.X.a);
		ConstraintLinesParallelism cnr1 = new ConstraintLinesParallelism(line4.p2,line4.p1,line2.p2,line2.p1);
		ConstraintLinesParallelism cnr2 = new ConstraintLinesParallelism(line3.p2,line3.p1,line1.p2,line1.p1);
		
		ConstraintTangency tang1 = new ConstraintTangency(line2.p2,line2.p1,cl.p1,cl.p2);
		ConstraintTangency tang2 = new ConstraintTangency(line4.p1,line4.p2,cl.p1,cl.p2);
		ConstraintTangency tang3 = new ConstraintTangency(line1.p1,line1.p2,cl.p1,cl.p2);
		//ConstraintTangency tang4 = new ConstraintTangency(line3.p1,line3.p2,cl.p1,cl.p2);

		ConstraintLinesSameLength sml = new  ConstraintLinesSameLength(line1.p1,line1.p2,line2.p1,line2.p2);
		
		
		//ConstraintDistance2Points con3 = new ConstraintDistance2Points(line1.p1,line1.p2 ,new Parameter(45));
		ConstraintDistance2Points con3 = new ConstraintDistance2Points(cl.p1,cl.p2 ,new Parameter(15));
		ConstraintDistance2Points con4 = new ConstraintDistance2Points(line5.p1,line5.p2 ,new Parameter(75));
		
		
		//trojkat + okreag
		ConstraintTangency t1 = new ConstraintTangency(line5.p1,line5.p2,c2.p1,c2.p2);
		ConstraintTangency t2 = new ConstraintTangency(line6.p1,line6.p2,c2.p1,c2.p2);
		ConstraintTangency t3 = new ConstraintTangency(line7.p1,line7.p2,c2.p1,c2.p2);

		//ConstraintLinesParallelism tpar2 = new ConstraintLinesParallelism(line7.p2,line7.p1,line2.p2,line2.p1);
		ConstraintLinesParallelism tpar3 = new ConstraintLinesParallelism(line5.p2,line5.p1,line1.p2,line1.p1);
		
		//Linia na lini - coincidence - dwa wiezy potrzebne
		//ConstraintLinesParallelism tpar4 = new ConstraintLinesParallelism(line2.p1,line2.p2,line7.p2,line2.p1); // punkt na lini
		ConstraintLinesParallelism tpar5 = new ConstraintLinesParallelism(line7.p1,line7.p2,line2.p1,line2.p2);
		
		
		//wiez dla trojkata na kat
		ConstraintAngle2Lines angelC = new ConstraintAngle2Lines(line5.p1,line5.p2,line6.p2,line6.p1,new Parameter(Math.PI/6));//30stopni
		
		//FIXME - powyzej trzeba jakos sprawdzac czy przypadkiem nie zadeklarowalismy zbyt duzo KATOW pomiedzy liniami, ale jak ?? 
		
		//System.out.println(tang.getHessian(Point.dbPoint, Parameter.dbParameter));
		/**
		 * Dlaczego wolniej zbiega sie dla wiezu prostopadlosci pomiedzy 
		 * liniami parametrycznymi ??
		 * Poniewaz zapomnialem zaimplementowac d(dFi/dq)'*lambda)/dq - czyli
		 * ta dodaktowa macierz - HESSIAN drugie pochodne
		 */
		//teraz wyswietlamy
		
		//System.out.println(GeometricPrymitive.dbPrimitives);
		//System.out.println(Constraint.dbConstraint);
		//System.out.println(Point.dbPoint);
		
		System.out.println("Wymiar zadania:"  + Point.dbPoint.size()*2 );
		System.out.println("Mnozniki Lagrange'a :" + Constraint.allLagrangeSize());
		System.out.println("Stopnie swobody : " + (Point.dbPoint.size()*2 -  Constraint.allLagrangeSize()));
		
		
		// Tworzymy Macierz "A" - dla tego zadania stala w czasie
		int sizeA = Point.dbPoint.size()*2 + Constraint.allLagrangeSize();
		MatrixDouble A= MatrixDouble.fill(sizeA,sizeA,0.0);
		MatrixDouble Fq = GeometricPrymitive.getAllForceJacobian();
		MatrixDouble Wq =null;//Constraint.getFullJacobian(Point.dbPoint, Parameter.dbParameter);
		//A.addSubMatrix(0, 0, Fq);
		//A.addSubMatrix(Fq.getHeight(), 0, Wq);
		//A.addSubMatrix(0, Fq.getWeight(), Wq.transpose()); 
		BindMatrix mA = null;
		//System.out.println("Rank + " + mA.rank());
		
		// Tworzymy wektor prawych stron b
		MatrixDouble b= null;
		BindMatrix mb = null;
		BindMatrix dmx = null;
		
		BindMatrix bmX = new BindMatrix(Point.dbPoint.size()*2 + Constraint.allLagrangeSize(),1);
		bmX.bind(Point.dbPoint);
	
		//System.out.println(bmX);
		//2 3 iteracje i jest git
		/** Liczba do skalowania wektora dx aby przyspieszyc obliczenia*/
		for(int i=0;i<10;i++){
			//zerujemy macierz A
			A= MatrixDouble.fill(sizeA,sizeA,0.0);
			//tworzymy macierz vector b
			b=MatrixDouble.mergeByColumn(GeometricPrymitive.getAllForce(),Constraint.getFullConstraintValues(Point.dbPoint, Parameter.dbParameter));
			b.dot(-1);
			//System.out.println(b);
			mb= new BindMatrix(b.m);
			
			// JACOBIAN
			Wq = Constraint.getFullJacobian(Point.dbPoint, Parameter.dbParameter);
			//HESSIAN
			A.addSubMatrix(0, 0, Fq.addC(Constraint.getFullHessian(Point.dbPoint, Parameter.dbParameter, bmX)));
			//A.addSubMatrix(0, 0, MatrixDouble.diagonal(Fq.getHeight(), 1.0)); // macierz diagonalna
			
			A.addSubMatrix(Fq.getHeight(), 0, Wq);
			A.addSubMatrix(0, Fq.getWeight(), Wq.transpose()); 
			mA = new BindMatrix(A.m);
			//System.out.println("Rank + " + mA.rank());
			// rozwiazjemy zadanie A*dx=b
			dmx = new BindMatrix(mA.solve(mb).getArray());
			
			// jezeli chcemy symulowac na bierzaco jak sie zmieniaja wiezy to 
			// wstawiamy jakis faktor dmx.times(0.3) , i<12
			bmX.plusEquals(dmx);
			bmX.copyToPoints();//uaktualniamy punkty
			
			//System.out.println("Wartosc wiezow : " + Constraint.getFullConstraintValues(Point.dbPoint, Parameter.dbParameter).transposeC());
		
			Matrix nrm = new Matrix(Constraint.getFullConstraintValues(Point.dbPoint, Parameter.dbParameter).getArray());
			System.out.println(" \n" + i + " : "+ nrm.norm1() + "\t" + nrm.norm2() + "\t" + nrm.normF() + "\n");
		}
		//System.out.println(Point.dbPoint);
		//System.out.println(Constraint.dbConstraint);
		//System.out.println(cn2.getValue(Point.dbPoint, Parameter.dbParameter));
		//System.out.println(GeometricPrymitive.dbPrimitives);
		//System.out.println(Constraint.dbConstraint);
		dbPrimitives =GeometricPrymitive.dbPrimitives;
		//Teraz wyswietlmy wiezy
		System.out.println(Constraint.dbConstraint);
		
		System.out.println(c2);
		// A na koniec relaksacja sprezyn 
		
	}
	
	@Override
	public void paint(Graphics g) {
		super.paint(g);
		Graphics2D g2 = (Graphics2D)g;
		//g2.scale(20, 20);
		g2.translate(150, 380);
		g2.scale(1, -1);

		g2.setColor(Color.ORANGE);
		g2.setStroke(new BasicStroke(1));
		g2.drawLine(0,0,300,0);//x
		g2.drawLine(0,0,0,300);//y
		int h=6;
		int k=4;//4
		Line l =null;
		Circle c =null;
		for(Integer i :dbPrimitives.keySet()){
			GeometricPrymitive gm = dbPrimitives.get(i);
			if(gm.type==GeometricPrymitiveType.Line){		
				l= (Line)gm;
				//p1 - p2
				g2.setStroke(new BasicStroke(2));
				g2.setColor(Color.BLACK);
				g2.draw(new Line2D.Double(l.p1.getX()*k, l.p1.getY()*k,l.p2.getX()*k,l.p2.getY()*k));
				//p1
				g2.draw(new Ellipse2D.Double(l.p1.getX()*k-h/2,l.p1.getY()*k-h/2,h,h));
				//p2
				g2.draw(new Ellipse2D.Double(l.p2.getX()*k-h/2,l.p2.getY()*k-h/2,h,h));
				
				/**
				g2.setColor(Color.BLUE);
				g2.setStroke(new BasicStroke(1));		
				g2.draw(new Line2D.Double(l.p1.getX()*k, l.p1.getY()*k, l.a.getX()*k,l.a.getY()*k));
				g2.draw(new Line2D.Double(l.p2.getX()*k, l.p2.getY()*k, l.b.getX()*k,l.b.getY()*k));
				*/
				
			}else if(gm.type==GeometricPrymitiveType.Circle){
				c= (Circle)gm;
				//p1 - p2
				g2.setStroke(new BasicStroke(1));
				g2.setColor(Color.BLACK);
				g2.draw(new Line2D.Double( c.p1.getX()*k, c.p1.getY()*k, c.p2.getX()*k,c.p2.getY()*k));
				
				g2.setStroke(new BasicStroke(2));
				//duzy okrag
				double radius = c.p2.sub(c.p1).length()*2;
				g2.draw(new Ellipse2D.Double((c.p1.x-radius/2)*k,(c.p1.y-radius/2)*k,radius*k,radius*k));
				//p1
				g2.setStroke(new BasicStroke(1));
				g2.draw(new Ellipse2D.Double(c.p1.getX()*k-h/2,c.p1.getY()*k-h/2,h,h));
				//p2
				g2.draw(new Ellipse2D.Double(c.p2.getX()*k-h/2,c.p2.getY()*k-h/2,h,h));
				
				/**
				g2.setColor(Color.GREEN);
				g2.setStroke(new BasicStroke(1));		
				g2.draw(new Line2D.Double(Math.floor(c.p1.getX()*k), Math.floor(c.p1.getY()*k), Math.floor(c.a.getX()*k),Math.floor(c.a.getY()*k)));
				g2.draw(new Line2D.Double(Math.floor(c.p2.getX()*k), Math.floor(c.p2.getY()*k), Math.floor(c.b.getX()*k),Math.floor(c.b.getY()*k)));
				*/	
			}else if(gm.type==GeometricPrymitiveType.Arc){
				
			}
			
		}
		//osie X i Y
		//System.out.println(GeometricPrymitive.dbPrimitives);
		
	}

}
