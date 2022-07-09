package com.mstruzek.sparsematrixsolver;

import com.mstruzek.msketch.matrix.MatrixDouble;


/**
 * Podstawowy wektor uzywany w systemie
 * 
 * @author root
 *
 */
public class BasicVector {
	
	/** dlugosc wektora */
	public int size;
	/** array z danymi */
	public double[] d =null;
	
	/**
	 * Podstawowy konstruktor do tworzenia vektor�w
	 * @param size
	 */
	public BasicVector(int size){
		this.size = size;
		d = new double[size];
		clear();
	}

	public BasicVector(BasicVector bv){
		this.size = bv.size;
		d = new double[size];
		for(int i=0;i<size;i++){
			d[i]=bv.d[i];
		}
		
	}
	/**
	 * Konstruktor na podstawie macierzy MatrixDouble
	 * tylko wektory kolumnowe
	 * @param md
	 */
	public BasicVector(MatrixDouble md){
	
		
		this.size = md.height();
		d = new double[size];
		for(int i=0;i<size;i++){
			d[i]=md.m[i][0];
		}
		
	}
	public static BasicVector merge(BasicVector... vectors){
		
		int fullSize = 0;
		
		for(int i=0;i<vectors.length;i++){
			fullSize +=vectors[i].size;
		}
		
		BasicVector out = new BasicVector(fullSize);
		
		int currentRow =0;
		for(int i=0;i<vectors.length;i++){
			
			System.arraycopy(vectors[i].d, 0, out.d, currentRow, vectors[i].size);

			currentRow+=vectors[i].size;
		}
		
		return out;
		
	}
	
	/**
	 * norma wektora -dlugosc
	 * @return
	 */
	public double norm(){
		
		double norm = 0;
		
		for(int i=0;i<size;i++){
			norm+=d[i]*d[i];
		}
		
		norm = Math.sqrt(norm);
		
		return norm;
	}
	
	/** 
	 * Zwraca losowy wektor o podanym rozmiarze
	 * @param size
	 * @return
	 */
	public static BasicVector vectorRandomFactory(int size){
		
		BasicVector v = new BasicVector(size);
		
		for(int i=0;i<size;i++){
			v.d[i]= Math.random();
		}
		
		return v;	
	}
	
	/**
	 * Iloczn skalarny dwoch wektor�w
	 * @param bv
	 * @return
	 */
	public double dot(BasicVector bv){
		
		double out = 0.0;
			
		for(int i=0;i<size;i++){
			out+=this.d[i]*bv.d[i];
		}
		return out;
	}
	
	/**
	 * Mnozenie razy skalar
	 * @param bv
	 * @return
	 */
	public BasicVector dot(double bv){
		
		BasicVector out = new BasicVector(size);
			
		for(int i=0;i<size;i++){
			out.d[i]=this.d[i]*bv;
		}
		return out;
	}	
	/**
	 * Dodaje dwa wektory
	 * c= this+bv
	 * @param bv
	 * @return c 
	 */
	public BasicVector addC(BasicVector bv){
		
		BasicVector out = new BasicVector(size);
			
		for(int i=0;i<size;i++){
			out.d[i]=this.d[i]+bv.d[i];
		}
		return out;
	}
	
	/**
	 * Dodaje dwa wektory
	 * this= this+bv
	 * @param bv
	 * @return c 
	 */
	public BasicVector add(BasicVector bv){
			
		for(int i=0;i<size;i++){
			this.d[i]=this.d[i]+bv.d[i];
		}
		return this;
	}
	
	/**
	 * Odejmuje 2 wektory
	 * c= this-bv
	 * @param bv
	 * @return c 
	 */
	public BasicVector sub(BasicVector bv){
		
		BasicVector out = new BasicVector(size);
			
		for(int i=0;i<size;i++){
			out.d[i]=this.d[i]-bv.d[i];
		}
		return out;
	}
	
	/**
	 * ustaw na zerow wszyskite wartosci
	 */
	public void clear(){
		for(int i=0;i<size;i++){
			d[i]=0.0;
		}	
	}
	
	public String toString(){
		String out = new String();
		
		out += " BasicVector : [ ";
		for(int i =0;i<size;i++){
			out += this.d[i] + " , ";
			if(( (i+1) % 4 ) == 0) out+="\n";
		}
		out+= "] ";
		return out;
		
	}
	
	public static void main(String[] args){
		BasicVector bv = BasicVector.vectorRandomFactory(3);
		BasicVector bv2 = BasicVector.vectorRandomFactory(3);
		BasicVector bv3 = BasicVector.vectorRandomFactory(3);
		System.out.println(bv + " norm : " + bv.norm());
		System.out.println(bv2 + " norm : " + bv.norm());
		System.out.println(bv3 + " norm : " + bv.norm());
		System.out.println(BasicVector.merge(bv,bv3,bv2) );
		
		
		MatrixDouble force = MatrixDouble.fill(8,1,2.0);
		BasicVector bs = new BasicVector(force);
		System.out.println(bs + " norm : " + bs.norm());
	}
}
