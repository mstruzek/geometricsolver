package com.mstruzek.sparsematrixsolver;

/**
 * Conjugate Gradient Method
 * 
 * A*x=b
 * @author root
 *
 */
public class CGMSolver {
	
	/** Macierz na ktorej pracujemy */
	MixedMatrix workMatrix;
	/** Wektor prawych stron w r�wnaniu */
	BasicVector b;
	
	/** Wektor wynikowy */
	BasicVector x;
	
	/** Residual Vector */
	BasicVector r;
	
	/** Residual vector w chwili nastepnej */
	BasicVector rk;
	
	/** pomocniczy */
	BasicVector pk;
	
	/** Wspolczynnik tymczasowy */
	double alfa =0.0;
	/** Wspolczynnik tymczasowy */
	double beta= 0.0;
	
	/** Norma */
	double norm = 1e-4;
	
	/** Ostatnia liczba iteracji */
	int iterations=0;
	private BasicVector x0;
	
	public CGMSolver(MixedMatrix workMatrix, BasicVector b,BasicVector x0) {
		super();
		this.workMatrix = workMatrix;
		this.b = b;
		this.x0 = x0;
		
		x= new BasicVector(b.size);
		
	}

	/**
	 * Funkcja rozwiazuje A*x=b
	 * @return x
	 */
	public BasicVector solve(){
		
		r=new BasicVector(b.sub(workMatrix.multiply(x0)));
		pk=new BasicVector(r);
		
		for(iterations=0;iterations<400;iterations++){
			alfa=r.dot(r)/pk.dot(workMatrix.multiply(pk));
			x.add(pk.dot(alfa));
			rk = r.addC(workMatrix.multiply(pk).dot(-alfa));
			if(r.dot(r)<norm) break;
			beta=rk.dot(rk)/r.dot(r);
			pk = rk.addC(pk.dot(beta));
			//przepisanie
			r=rk;
		}
		
		return x;
	}
	
	public static void main(String[] args) {
		int sz=40;
		
		MixedMatrix mm = new MixedMatrix(sz,sz);
		
		DenseMatrix dm= DenseMatrix.matrixRandomFactory(sz);
		
		SparseMatrix sm = new SparseMatrix(sz,sz);
		sm.set(0, 0, dm);
		mm.addSubMatrix(new MatrixAnchor(0,0,sm));
		
		//System.out.println(mm);
		
		BasicVector xn = BasicVector.vectorRandomFactory(sz);
		BasicVector b = mm.multiply(xn);
		BasicVector x0 = new BasicVector(b.size);
		
		//System.out.println(b);
		CGMSolver cgm = new CGMSolver(mm,b,x0);
		
		long start = System.currentTimeMillis(); // start timing
		
		BasicVector x = cgm.solve();
		
        long stop = System.currentTimeMillis(); // stop timing
        System.out.println("TimeMillis: " + (stop - start) + " - Iterations :" +  cgm.iterations); // print execution time
        
        
		//System.out.println(xn);
		//System.out.println(x);
        
		BasicVector res = xn.sub(x);
		
		System.out.println("Residual :" + res.dot(res));
	}
	
}
