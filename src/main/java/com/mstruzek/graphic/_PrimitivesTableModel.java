package com.mstruzek.graphic;

import java.util.ArrayList;
import javax.swing.table.AbstractTableModel;

import com.mstruzek.msketch.GeometricPrimitive;

public class _PrimitivesTableModel extends AbstractTableModel implements TableModelRemovable{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	/** zmienna w ktorej przechowujemy wszystkie wiezy nalozene przez uzytkownika */
	ArrayList<GeometricPrimitive> primitivesContainer = new ArrayList<GeometricPrimitive>();
	
	private String[] columnNames = {"id","Type","p1","p2","p3"};
	
	
	_PrimitivesTableModel(){
		super();
		
	}
	
	public void add(GeometricPrimitive c){
		primitivesContainer.add(c);
		fireTableRowsInserted(primitivesContainer.size(),primitivesContainer.size());
	}
	/**
	 * Usun elemt z modelu i z globalnego pojemnika
	 * @param i - numer w constraintContainer
	 */
	public void remove(int i){
		if(i<0) return;
		int id=primitivesContainer.get(i).getPrimitiveId();
		primitivesContainer.remove(i);
		GeometricPrimitive.dbPrimitives.remove(id);
		fireTableRowsDeleted(i, i);
	}
	
	@Override
	public int getColumnCount() {

		return 5;
	}

	@Override
	public int getRowCount() {
		return primitivesContainer.size();
	}

	@Override
	public Object getValueAt(int wiersz, int kolumna) {
		
		int out;

		switch(kolumna){
		case 0:
			//return "id";
			return primitivesContainer.get(wiersz).getPrimitiveId();
		case 1:
			//return "type";
			return primitivesContainer.get(wiersz).getType();
		case 2:
			return primitivesContainer.get(wiersz).getP1();
		case 3:
			//return "L";	
			out = primitivesContainer.get(wiersz).getP2();
			if(out==-1) return null;
			else return out;
		case 4:
			//return "M";
			out = primitivesContainer.get(wiersz).getP3();
			if(out==-1) return null;
			else return out;

		}
		return null;
	}
	
	public String getColumnName(int col) {
		return columnNames[col];        
    }

	 public Class getColumnClass(int c) {
	        //return getValueAt(0, c).getClass();
	        return String.class;
	 }
	 
	 public boolean isCellEditable(int row, int col){
		return  false;
	 }

}

