package com.mstruzek.controller;

import com.mstruzek.msketch.Model;
import com.mstruzek.graphic.MyTableModel;
import com.mstruzek.msketch.Parameter;

public class ParametersTableModel extends MyTableModel {

    private static final String[] PARAMETERS_COLUMN_NAMES = {"id", "value"};


    @Override
    public void remove(int i) {
        if(i < 0) return;
        int id = Model.parameterContainer.get(i).getId();
        Model.parameterContainer.remove(i);
        Parameter.dbParameter.remove(id);
        fireTableRowsDeleted(i, i);
    }

    @Override
    public int getColumnCount() {
        return 2;
    }

    @Override
    public int getRowCount() {
        return Model.parameterContainer.size();
    }

    @Override
    public Object getValueAt(int rowIndex, int columnIndex) {

        switch (columnIndex) {
            case 0:
                //return "id";
                return Model.parameterContainer.get(rowIndex).getId();
            case 1:
                //return "type";
                return Model.parameterContainer.get(rowIndex).getValue();
        }
        return null;

    }

    public String getColumnName(int col) {
        return PARAMETERS_COLUMN_NAMES[col];
    }

    public Class getColumnClass(int c) {
        return String.class;
    }

    public boolean isCellEditable(int row, int col) {
        if(col > 0) return true;
        return false;
    }

    public void setValueAt(Object value, int row, int col) {
        double d = Double.parseDouble(value.toString());
        if(col > 0) {
            Model.parameterContainer.get(row).setValue(d);
        }
        fireTableCellUpdated(row, col);
    }
}
