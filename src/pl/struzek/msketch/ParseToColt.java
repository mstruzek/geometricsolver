package pl.struzek.msketch;

import pl.struzek.msketch.matrix.BindMatrix;
import pl.struzek.msketch.matrix.MatrixDouble;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;

/**
 * Klasa zawiera same funkcje static
 * do przeformatowania macierzy gestych na macierze rzadkie 
 * @author root
 *
 */
public class ParseToColt {
	
	
	/**
	 * Konwersja macierzy gestej do macierzy rzadkiej
	 * @param md
	 * @return
	 */
	public static SparseDoubleMatrix2D toSparse(MatrixDouble md){
		
		SparseDoubleMatrix2D matrix2D = new SparseDoubleMatrix2D(md.getHeight(),md.getWeight());
		
		for(int i=0;i<md.getHeight();i++){
			for(int j=0;j<md.getWeight();j++){
				if(md.m[i][j]!=0.0){
					matrix2D.setQuick(i, j, md.m[i][j]);
				}
			}
		}
		return matrix2D;
		
	}

	public static DoubleMatrix1D toDenseVector(MatrixDouble b) {
		DoubleMatrix1D doubleMatrix1D = new DenseDoubleMatrix1D(b.getHeight());
		
		for(int i=0;i<b.getHeight();i++){
			doubleMatrix1D.setQuick(i, b.m[i][0]);
		}
		
		return doubleMatrix1D;
	}

	public static BindMatrix toBindVector(DoubleMatrix1D matrix1Db) {
		
		BindMatrix dmx= new BindMatrix(matrix1Db.size(),1);
		
		for(int i=0;i<matrix1Db.size();i++){
			dmx.set(i, 0, matrix1Db.getQuick(i));
		}
		
		return dmx;
		
	}
	
	

}
