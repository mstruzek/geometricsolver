package com.mstruzek.graphic;

import java.util.ArrayList;
import javax.swing.table.AbstractTableModel;

import com.mstruzek.msketch.Constraint;

public class _ConstraintTableModel extends AbstractTableModel implements TableModelRemovable{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	/** zmienna w ktorej przechowujemy wszystkie wiezy nalozene przez uzytkownika */
	ArrayList<Constraint> constraintContainer = new ArrayList<Constraint>();
	
	private String[] columnNames = {"id","Type","K","L","M","N","P-id"};
	
	
	_ConstraintTableModel(){
		super();
		
	}
	
	public void add(Constraint c){
		constraintContainer.add(c);
		fireTableRowsInserted(constraintContainer.size(), constraintContainer.size());
	}
	/**
	 * Usun elemt z modelu i z globalnego pojemnika
	 * @param i - numer w constraintContainer
	 */
	public void remove(int i){
		if(i<0) return;
		int id=constraintContainer.get(i).getConstraintId();
		constraintContainer.remove(i);
		Constraint.dbConstraint.remove(id);
		fireTableRowsDeleted(i, i);
	}
	
	@Override
	public int getColumnCount() {

		return 7;
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
			return constraintContainer.get(wiersz).getConstraintId();
		case 1:
			//return "type";
			return constraintContainer.get(wiersz).getConstraintType();
		case 2:
			//return "K";
			return constraintContainer.get(wiersz).getK();
		case 3:
			//return "L";	
			return constraintContainer.get(wiersz).getL();
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
