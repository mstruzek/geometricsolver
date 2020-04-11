package com.mstruzek.sparsematrixsolver;

import java.util.ArrayList;

/**
 * Macierz do przechowywania macierzy rzadkich i pelnych
 * 
 * @author root
 *
 */
public class MixedMatrix {

	/** Lista wszystkich kotwic do macierzy */
	ArrayList<MatrixAnchor> matrixList = new ArrayList<MatrixAnchor>();
	
	/** liczba wierszy */
	int height;
	/** liczba kolumn */
	int width;
	
	/**
	 * Tworzy nowa macierz o podanych wymiarach
	 * @param height ilosc wierszy
	 * @param width ilosc kolumn
	 */
	public MixedMatrix(int height, int width) {
		super();
		this.height = height;
		this.width = width;
	}

	/**
	 * Dodaje nowa podmacierz
	 * @param anchor
	 */
	public void addSubMatrix(MatrixAnchor anchor){

		//pierw powinno byc sprawdzenie czy macierz miesci sie w zakresie
		if( ( (anchor.columnPosition + anchor.matrixData.getWidth()) <=this.width) && ( (anchor.rowPosition+anchor.matrixData.getHeight())<=this.height) ){
			this.matrixList.add(anchor);
		}	
	}
	
	/**
	 * Mnozy wektor prze macierz
	 * @param inVector
	 * @return outVector
	 */
	public BasicVector multiply(BasicVector inVector){
		
		BasicVector outVector = new BasicVector(this.height);
		
		MatrixAnchor ma = null;
		for(int i=0;i<matrixList.size();i++){
			ma = matrixList.get(i);
			//FIXME - tu sie upewnic czy dobrze jest
			ma.matrixData.multiply(outVector,inVector,ma.rowPosition,ma.columnPosition);
		}
		return outVector;
	}
	/** 
	 * Zwraca ilosc wierszy
	 * @return
	 */
	public int getHeight() {
		return height;
	}

	/**
	 * Zwraca ilosc kolumn 
	 * @return
	 */
	public int getWidth() {
		return width;
	}

	public double get(int row ,int col){
		double out =0.0;
		
		MatrixAnchor ma = null;
		for(int i=0;i<matrixList.size();i++){
			ma = matrixList.get(i);
			if((row>=ma.rowPosition) && (col>=ma.columnPosition)){
				if((row<(ma.rowPosition+ma.matrixData.getHeight())) && (col<(ma.columnPosition+ma.matrixData.getWidth()))){
					out+=ma.matrixData.get(row-ma.rowPosition, col-ma.columnPosition);
				}
			}
		}
		return out;
	}
	
	public String toString(){
		String out = new String();
		for(int i=0;i<height;i++){
			for(int j=0;j<width;j++){
				out+=get(i, j) + "\t";
			}
			out+="\n";
		}
		return out;
		
	}
	public static void main(String[] args){

		MixedMatrix mm = new MixedMatrix(9,9);
		
		DenseMatrix dm= new DenseMatrix(4,4);
		for(int i=0;i<4;i++){
			for(int j=0;j<4;j++){
				dm.d[i][j]=(i+j);
			}
		}
		
		//System.out.println(dm);
		DenseMatrix dm2 = new DenseMatrix(2,2);
		for(int i=0;i<2;i++){
			for(int j=0;j<2;j++){
				dm2.d[i][j]=10;
			}
		}	
		
		SparseMatrix sm = new SparseMatrix(3,3);
		sm.set(0,0, 5);
		sm.set(0,2,2);
		sm.set(2,1,3);
		
		mm.addSubMatrix(new MatrixAnchor(0,0,dm));
		mm.addSubMatrix(new MatrixAnchor(4,4,dm2));
		mm.addSubMatrix(new MatrixAnchor(6,6,sm));
		//sm.multiply(2.0);
		//sm =(SparseMatrix) sm.transposeC();
		
		BasicVector bv = new BasicVector(9);
		bv.d[0] =1;		bv.d[1] =1;		bv.d[2] =1;		bv.d[3] =1;		bv.d[4] =1;
		bv.d[5] =1;		bv.d[6] =1;		bv.d[7] =1;		bv.d[8] =1;		
		
		System.out.println(bv);

		BasicVector out =mm.multiply(bv);
		System.out.println(out);
		System.out.println(mm);
	}	
	
}
