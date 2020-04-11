package com.mstruzek.graphic;

/** 
 * Dodatkowy interfejs w celu usuwania 
 * elementow z AbstractTbaleModel dla Constraint, Primitives,Parameters
 * @author root
 *
 */
public interface TableModelRemovable {
	
	/** funkcja usuwa dany element z list modelu */
	public void remove(int i);

}
