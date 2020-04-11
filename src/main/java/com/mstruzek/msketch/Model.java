package com.mstruzek.msketch;

import Jama.Matrix;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.LUDecompositionQuick;
import com.mstruzek.controller.ControllerInterface;
import com.mstruzek.graphic.TableModelRemovable;
import com.mstruzek.msketch.matrix.BindMatrix;
import com.mstruzek.msketch.matrix.MatrixDouble;

import javax.swing.table.AbstractTableModel;
import java.util.ArrayList;

/** 
 * Klasa w ktorej przechowujemy caly model
 * matematyczny naszego szkicownika
 * 
 * @author root
 *
 */
public class Model implements ControllerInterface {

	/** zmienna w ktorej przechowujemy wszystkie elementy geometryczne */
	ArrayList<GeometricPrymitive> primitivesContainer = new ArrayList<GeometricPrymitive>();
	private String[] primitivesColumnNames = {"id","Type","p1","p2","p3"};
	
	/** zmienna w ktorej przechowujemy wszystkie wiezy nalozene przez uzytkownika */
	ArrayList<Parameter> parametersContainer = new ArrayList<Parameter>();
	private String[] parametersColumnNames = {"id","value"};
	
	/** zmienna w ktorej przechowujemy wszystkie wiezy nalozene przez uzytkownika */
	ArrayList<Constraint> constraintContainer = new ArrayList<Constraint>();
	private String[] constraintColumnNames = {"Type","Type","K","L","M","N","P-id","Norm"};	
	
	
	MyTableModel constraintTableModel;
	
	MyTableModel primitivesTableModel;
	
	MyTableModel parametersTableModel;
	
	
	public Model() {
		super();
		
		getPrimitivesTableModel();
		getParametersTableModel();
		getConstraintTableModel();
		//addLine(new Vector(0,0), new Vector(50,50));
	}
	
	/**
	 * Rozwiaz model , przygotuj zadanie
	 */
	@Override
	public void solveSystem(){
		
		System.out.println("*************************");
        long start = System.currentTimeMillis(); // start timing
        
		//pierw uakutalnic wspolrzedne punktow w kazdym GeometricPrimiticves
		// relaxForces() ;
		
		//teraz obliczenia
		
		// System.out.println("Wymiar zadania:"  + Point.dbPoint.size()*2 );
		// System.out.println("Mnozniki Lagrange'a :" + Constraint.allLagrangeSize());
		// System.out.println("Stopnie swobody : " + (Point.dbPoint.size()*2 -  Constraint.allLagrangeSize()));
		
		if(Point.dbPoint.size()==0) return;
		
		// Tworzymy Macierz "A" - dla tego zadania stala w czasie
		int sizeA = Point.dbPoint.size()*2 + Constraint.allLagrangeSize();
		MatrixDouble A= MatrixDouble.fill(sizeA,sizeA,0.0);
		MatrixDouble Fq = GeometricPrymitive.getAllForceJacobian();

		
		MatrixDouble Wq =null;//Constraint.getFullJacobian(Point.dbPoint, Parameter.dbParameter);

		BindMatrix mA = null;
		
		// Tworzymy wektor prawych stron b
		MatrixDouble b= null;
		BindMatrix mb = null;
		BindMatrix dmx = null;
		
		BindMatrix bmX = new BindMatrix(Point.dbPoint.size()*2 + Constraint.allLagrangeSize(),1);
		bmX.bind(Point.dbPoint);
		
		
		//FIXME - jezeli po 6 iteracjach problem ze zbieznoscia warto przeliczyc punkty kontrolne
		// czyli przeliczyc punkty i nowe Fq - macierz obliczyc

		double erri1,erri = 0,delta;
		System.out.println(" Iter/Time [ms] /Norm ");
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
			

			//dmx = new BindMatrix(mA.solve(mb).getArray());
			
	
			/** WS*/ // JEST PRZYSPIESZENIE WYRABIA SIE w 380 ms dla 20 lini gdzie 100ms na jedna decompozycje
			
			
			
			DoubleMatrix2D matrix2DA = ParseToColt.toSparse(A);

			
			DoubleMatrix1D matrix1Db = ParseToColt.toDenseVector(b);
			
			//System.out.println(matrix2DA.cardinality() + " : " + (matrix1Db.size()*matrix1Db.size()));
			
			//System.out.println(matrix2DA);
			//System.out.println(A);
			long start2 = System.currentTimeMillis(); // start timing
			LUDecompositionQuick lu = new LUDecompositionQuick();
			lu.decompose(matrix2DA);
			lu.solve(matrix1Db);
			long stop2 = System.currentTimeMillis(); // stop timing
			dmx =ParseToColt.toBindVector(matrix1Db);
			
	        
	        long deltat2 = stop2-start2;
	        
	        //System.out.println("TimeMillis2: " + deltat2); // print execution time
			
			/** KONIEC  WS*/

			
			// jezeli chcemy symulowac na bierzaco jak sie zmieniaja wiezy to 
			// wstawiamy jakis faktor dmx.times(0.3) , i<12
			bmX.plusEquals(dmx);
			bmX.copyToPoints();//uaktualniamy punkty
			//System.out.println(bmX);
		
			Matrix nrm = new Matrix(Constraint.getFullConstraintValues(Point.dbPoint, Parameter.dbParameter).getArray());
			System.out.println(" \n " + (i+1) + " || " + deltat2 + "  ||  " + nrm.norm1() +  "\n");
			
			
			//stary warunek wyjscia
			if(nrm.norm1()<0.05){
				//System.out.println("New Norm + :" + Constraint.getFullNorm(Point.dbPoint, Parameter.dbParameter));
				break;
			}
			
			//Matrix nforce = new Matrix(GeometricPrymitive.getAllForce().getArray());
			//System.out.println( "Force " + i + " - " + nforce.norm1());
			//minumalizujmy sily
			
			
			if(i==0){
				erri = nrm.norm1();
			}
			
			//liczymy zmiane bledu
			if(i>0){
				erri1 = nrm.norm1();
				delta = erri1-erri;
				erri = erri1;
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

        long stop = System.currentTimeMillis(); // stop timing
        long deltat = stop-start;
        
        System.out.println("TimeMillis: " + deltat); // print execution time
        
		//System.out.println(Point.dbPoint);
		
	}

	/** 
	 * Zwraca widok na model elementow geometrycznych
	 */
	public MyTableModel getPrimitivesTableModel(){
		
		primitivesTableModel = new MyTableModel(){

			private static final long serialVersionUID = 1L;

			@Override
			public void remove(int i) {
				if(i<0) return;
				int id=primitivesContainer.get(i).getPrimitiveId();
				
				//usun wiezy powiazane
				int[] con = primitivesContainer.get(i).associateConstraintsId;
				for(int k=0;k<con.length;k++){
					Constraint.dbConstraint.remove(con[k]);
				}
				//usun punkty
				for(int pi:primitivesContainer.get(i).getAllPointsId()){
					Point.dbPoint.remove(pi);
				}
				
				primitivesContainer.remove(i);
				GeometricPrymitive.dbPrimitives.remove(id);
				fireTableRowsDeleted(i, i);
				
				//FIXME -JESZCE PUNKTY TRZEBA USUNAC 
				System.out.println(GeometricPrymitive.dbPrimitives);
				System.out.println(Constraint.dbConstraint);
			}

			@Override
			public int getColumnCount() {
				return 2;
			}

			@Override
			public int getRowCount() {
				return primitivesContainer.size();
			}

			@Override
			public Object getValueAt(int rowIndex, int columnIndex) {
				
				int out;

				switch(columnIndex){
				case 0:
					//return "id";
					return primitivesContainer.get(rowIndex).getPrimitiveId();
				case 1:
					//return "type";
					return primitivesContainer.get(rowIndex).getType();
				case 2:
					return primitivesContainer.get(rowIndex).getP1();
				case 3:
					//return "L";	
					out = primitivesContainer.get(rowIndex).getP2();
					if(out==-1) return null;
					else return out;
				case 4:
					//return "M";
					out = primitivesContainer.get(rowIndex).getP3();
					if(out==-1) return null;
					else return out;

				}
				return null;
			}
			public String getColumnName(int col) {
				return primitivesColumnNames[col];        
		    }	
			
			public Class getColumnClass(int c) {
				return String.class;
			}
			 
			public boolean isCellEditable(int row, int col){
				return  false;
			}
			
		};
		return primitivesTableModel;
	}

	/** 
	 * Zwraca widok na parametry
	 */	
	public MyTableModel getParametersTableModel(){
		
		parametersTableModel= new MyTableModel(){

			private static final long serialVersionUID = 1L;

			@Override
			public void remove(int i) {
				if(i<0) return;
				int id=parametersContainer.get(i).getId();
				parametersContainer.remove(i);
				Parameter.dbParameter.remove(id);
				fireTableRowsDeleted(i, i);			
			}

			@Override
			public int getColumnCount() {
				return 2;
			}

			@Override
			public int getRowCount() {
				return parametersContainer.size();
			}

			@Override
			public Object getValueAt(int rowIndex, int columnIndex) {
				
				switch(columnIndex){
				case 0:
					//return "id";
					return parametersContainer.get(rowIndex).getId();
				case 1:
					//return "type";
					return parametersContainer.get(rowIndex).getValue();
				}
				return null;
				
			}
			public String getColumnName(int col) {
				return parametersColumnNames[col];          
		    }	
			
			public Class getColumnClass(int c) {
				return String.class;
			}
			 
			public boolean isCellEditable(int row, int col){
				if(col>0) return true;
				return false;
			}	
			
			public void setValueAt(Object value, int row, int col) {
				double d = Double.parseDouble(value.toString());
				if(col>0){
					parametersContainer.get(row).setValue(d);
				}
				fireTableCellUpdated(row, col);
			}			
		};	
		return parametersTableModel;
	}

	
	/**
	 * Zwraca widok na model wiezow dodanych przez uzytkownika
	 */
	public MyTableModel getConstraintTableModel(){
		constraintTableModel =  new MyTableModel(){
			
			public void remove(int i){
				if(i<0) return;
				int id=constraintContainer.get(i).getConstraintId();
				int parId = constraintContainer.get(i).getParametr();
				if(parId>=0){
					for(int k=0;k<parametersContainer.size();k++){
						if(parametersContainer.get(k).getId()==parId){
							parametersContainer.remove(k);
							Parameter.dbParameter.remove(parId);
							parametersTableModel.fireTableRowsDeleted(k, k);	
						}
							
					}
				}
				constraintContainer.remove(i);
				Constraint.dbConstraint.remove(id);
				fireTableRowsDeleted(i, i);
				
			}
			
			@Override
			public int getColumnCount() {

				return 1;
			}

			@Override
			public int getRowCount() {
				return constraintContainer.size();
			}

			@Override
			public Object getValueAt(int wiersz, int kolumna) {
				
				int out;

				switch(kolumna){
				case 0:
					//return "id";
					//return constraintContainer.get(wiersz).getConstraintId();
					return constraintContainer.get(wiersz).getConstraintType();
				case 1:
					//return "type";
					return constraintContainer.get(wiersz).getConstraintType();
				case 2:
					//return "K";
					return constraintContainer.get(wiersz).getK();
				case 3:
					//return "L";	
					out =constraintContainer.get(wiersz).getL();
					if(out==-1) return null;
					else return out;
				case 4:
					//return "M";
					out =constraintContainer.get(wiersz).getM();
					if(out==-1) return null;
					else return out;
				case 5:
					//return "N";
					out =constraintContainer.get(wiersz).getN();
					if(out==-1) return null;
					else return out;
				case 6:
					//return "P-id";	
					out =constraintContainer.get(wiersz).getParametr();
					if(out==-1) return null;
					else return out;
				case 7:
					return constraintContainer.get(wiersz).getNorm(Point.dbPoint, Parameter.dbParameter);
				}
				return null;
			}
			
			public String getColumnName(int col) {
				return constraintColumnNames[col];        
		    }

			 public Class getColumnClass(int c) {
			        //return getValueAt(0, c).getClass();
			        return String.class;
			 }
			 
			 public boolean isCellEditable(int row, int col){
				return  false;
			 }
			
		};
		return constraintTableModel;
	}

	
	@Override
	public void addLine(Vector v1, Vector v2) {
		primitivesContainer.add(new Line(v1,v2));
		primitivesTableModel.fireTableRowsInserted(primitivesContainer.size(), primitivesContainer.size());
	}

	@Override
	public void addCircle(Vector v1, Vector v2) {
		primitivesContainer.add(new Circle(v1,v2));
		primitivesTableModel.fireTableRowsInserted(primitivesContainer.size(), primitivesContainer.size());	
	}

	@Override
	public void addArc(Vector v1, Vector v2) {
		primitivesContainer.add(new Arc(v1,v2,null));
		primitivesTableModel.fireTableRowsInserted(primitivesContainer.size(), primitivesContainer.size());	
	}

	@Override
	public void addPoint(Vector v1) {
		primitivesContainer.add(new FreePoint(v1));
		primitivesTableModel.fireTableRowsInserted(primitivesContainer.size(), primitivesContainer.size());	
		
	}
	

	@Override
	public void addConstraint(GeometricConstraintType constraintType, int K,int L, int M, int N, double p) {
		
		Point pK = null,pL = null,pM = null,pN = null;
		
		if(K>=0){
			pK = Point.dbPoint.get(K);
			pL = Point.dbPoint.get(L);
			pM = Point.dbPoint.get(M);
			pN = Point.dbPoint.get(N);
		}else return;
		
		Parameter par;

		switch(constraintType){
		case Conect2Points:
			constraintContainer.add(new ConstraintConect2Points(pK,pL));
			break;
		case FixPoint:
			constraintContainer.add(new ConstraintFixPoint(pK));
			break;
			
		case LinesPerpendicular:
			constraintContainer.add(new ConstraintLinesPerpendicular(pK,pL,pM,pN));
			break;
			
		case LinesParallelism:
			constraintContainer.add(new ConstraintLinesParallelism(pK,pL,pM,pN));

			break;
		case Tangency:
			constraintContainer.add(new ConstraintTangency(pK,pL,pM,pN));
			break;
			
		case Distance2Points:
			par = new Parameter(p);
			constraintContainer.add(new ConstraintDistance2Points(pK,pL,par));
			parametersContainer.add(par);
			parametersTableModel.fireTableRowsInserted(parametersContainer.size(), parametersContainer.size());			break;
		case Angle2Lines:
			par= new Parameter(p);
			constraintContainer.add(new ConstraintAngle2Lines(pK,pL,pM,pN,par));
			parametersContainer.add(par);
			parametersTableModel.fireTableRowsInserted(parametersContainer.size(), parametersContainer.size());
			break;
			
		case DistancePointLine:
			//FIXME - dokonczyc wiez
			par= new Parameter(p);
			constraintContainer.add(null);
			parametersContainer.add(par);
			parametersTableModel.fireTableRowsInserted(parametersContainer.size(), parametersContainer.size());			break;
			
		case LinesSameLength:
			constraintContainer.add(new ConstraintLinesSameLength(pK,pL,pM,pN));
			break;
			
		case SetVertical:
			constraintContainer.add(new ConstraintVertical(pK,pL));
			break;
		case SetHorizontal:
			constraintContainer.add(new ConstraintHorizontal(pK,pL));
			break;
		}
		constraintTableModel.fireTableRowsInserted(constraintContainer.size(), constraintContainer.size());		
	}

	public ArrayList<GeometricPrymitive> getPrimitivesContainer() {
		return primitivesContainer;
	}

	public abstract class MyTableModel extends AbstractTableModel implements TableModelRemovable {
		
		private static final long serialVersionUID = 1L;

		public MyTableModel() {
			super();
		}

	}

	@Override
	public void relaxForces() {
		for(Integer g:GeometricPrymitive.dbPrimitives.keySet()){
			GeometricPrymitive.dbPrimitives.get(g).recalculateControlPoints();
		}
		
	}

}
