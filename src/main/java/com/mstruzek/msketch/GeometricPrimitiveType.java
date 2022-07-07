package com.mstruzek.msketch;

/** 
 * Podstawowe elementy geometryczne ktore mozna narysowac 
 * */
public enum GeometricPrimitiveType{
	/**Punkt */
	FreePoint,
	/**Prosta */
	Line,
	/** "Sztywna" Linia - nie podlegajaca wiezom */
	FixLine,
	/** Okr�g */
	Circle,
	/** Cz�� okregu , luk */
	Arc; 
	
}
