package pl.struzek.graphic;

import java.util.ArrayList;
import javax.swing.table.AbstractTableModel;

import pl.struzek.msketch.Parameter;


public class  _ParametersTableModel extends AbstractTableModel implements TableModelRemovable{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	/** zmienna w ktorej przechowujemy wszystkie wiezy nalozene przez uzytkownika */
	ArrayList<Parameter> parametersContainer = new ArrayList<Parameter>();
	
	private String[] columnNames = {"id","value"};
	
	
	_ParametersTableModel(){
		super();
		
	}
	
	public void add(Parameter c){
		parametersContainer.add(c);
		fireTableRowsInserted(parametersContainer.size(),parametersContainer.size());
	}
	/**
	 * Usun elemt z modelu i z globalnego pojemnika
	 * @param i - numer w constraintContainer
	 */
	public void remove(int i){
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
	public Object getValueAt(int wiersz, int kolumna) {

		switch(kolumna){
		case 0:
			//return "id";
			return parametersContainer.get(wiersz).getId();
		case 1:
			//return "type";
			return parametersContainer.get(wiersz).getValue();
		}
		return null;
	}
	
	 public void setValueAt(Object value, int row, int col) {
		 double d = Double.parseDouble(value.toString());
	        if(col>0){
	        	parametersContainer.get(row).setValue(d);
	        }
	        fireTableCellUpdated(row, col);
	 }

	
	public String getColumnName(int col) {
		return columnNames[col];        
    }

	 public Class getColumnClass(int c) {
	        //return getValueAt(0, c).getClass();
	        return String.class;
	 }
	 
	 public boolean isCellEditable(int row, int col){
		if(col>0) return true;
		return false;
	 }

}

