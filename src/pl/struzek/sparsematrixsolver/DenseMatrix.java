package pl.struzek.sparsematrixsolver;

import pl.struzek.msketch.matrix.MatrixDouble;

/**
 * Klasa reprezentuje macierz "gêst¹"
 * @author root
 *
 */
public class DenseMatrix implements MatrixData {

	/** szerokosc */
	int width;
	/** wysokosc */
	int height;
	/** ty przechowuje macierz */
	double[][] d = null;
	
	public DenseMatrix(int width, int height) {
		super();
		this.width = width;
		this.height = height;
		d = new double[height][];
		for(int i=0;i<height;i++){
			d[i]= new double[width];
		}
	}
	
	public DenseMatrix(MatrixDouble md){
		super();
		this.width = md.getWeight();
		this.height = md.getHeight();
		d = new double[height][];
		for(int i=0;i<height;i++){
			d[i]= new double[width];
		}		
		
		for(int i=0;i<height;i++){
			for(int j=0;j<width;j++){
				d[i][j] = md.m[i][j];
			}
		}
	}

	@Override
	public void add(int row, int col, double val) {
		if((row<height) && (col<width))	d[row][col]+=val;
	}

	@Override
	public void set(int row, int col, double val) {
		if((row<height) && (col<width)) d[row][col]=val;	
	}

	@Override
	public double get(int row, int col) {
		if((row<height) && (col<width))	return d[row][col];
		return 0.0;
	}

	@Override
	public int getHeight() {
		return this.height;
	}

	@Override
	public int getWidth() {
		return this.width;
	}



	@Override
	public void multiply(double in) {
		for(int i=0;i<height;i++){
			for(int j=0;j<width;j++){
				d[i][j]=d[i][j]*in;
			}
		}
	}

	@Override
	public void multiply(BasicVector out, BasicVector in, int startRow,
			int startColumn) {
		
		for(int i=0;i<height;i++){
			for(int j=0;j<width;j++){
				out.d[i+startRow]+=d[i][j]*in.d[j+startColumn];
			}
		}
		
	}

	@Override
	public MatrixData transposeC() {
		DenseMatrix dm = new DenseMatrix(this.height,this.width);
		for(int i=0;i<height;i++){
			for(int j=0;j<width;j++){
				dm.d[j][i]=d[i][j];
			}
		}
		return dm;
	}
	
	public String toString(){
		String out = new String();
		for(int i=0;i<height;i++){
			for(int j=0;j<width;j++){
				out+=d[i][j] + "\t";
			}
			out+="\n";
		}
		return out;
		
	}
	/**
	 * Macierz jednostkowa przeskalowana
	 * @param factor
	 * @return
	 */
	public static DenseMatrix eye(int size,double factor){
		DenseMatrix dm = new DenseMatrix(size,size);
		for(int i=0;i<size;i++){
			dm.d[i][i]=factor;			
		}

		return dm;
	}
	
	public static DenseMatrix matrixRandomFactory(int size){
		
		DenseMatrix dm = new DenseMatrix(size,size);
		
		double tmp =0.0;
		for(int i=0;i<size;i++){
			for(int j=i;j<size;j++){
				tmp= Math.random();
				dm.d[i][j]=tmp;
				dm.d[j][i]=tmp;
			}
			
		}
		
		return dm;	
	}
	public static void main(String[] args){

		DenseMatrix dm= new DenseMatrix(4,4);
		for(int i=0;i<4;i++){
			for(int j=0;j<4;j++){
				dm.d[i][j]=(i+j);
			}
		}
		System.out.println(dm);
		
		BasicVector bv = new BasicVector(4);
		bv.d[0] =1;bv.d[1] =1;bv.d[2] =0;bv.d[3] =0;
		System.out.println(bv);
		
		BasicVector out = new BasicVector(4);
		dm.multiply(out, bv, 0, 0);
		System.out.println(out);
		
		MatrixDouble force = MatrixDouble.fill(8,2,3.0);
		
		DenseMatrix dq= new DenseMatrix(force);
		
		System.out.println(dq);
		
	}		
}
