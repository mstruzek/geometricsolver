package com.mstruzek.controller;

import com.mstruzek.msketch.Parameter;

import javax.swing.table.AbstractTableModel;

public class ParametersTableModel extends AbstractTableModel{

    private static final String[] PARAMETERS_COLUMN_NAMES = {"id", "value"};

    @Override
    public int getColumnCount() {
        return 2;
    }

    @Override
    public int getRowCount() {
        return Parameter.dbParameter.size();
    }

    @Override
    public Object getValueAt(int rowIndex, int columnIndex) {
        Parameter parameter=Parameter.dbParameter.values().toArray(new Parameter[]{})[rowIndex];
        switch (columnIndex) {
            case 0:
                return parameter.getId();
            case 1:
                return parameter.getValue();
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
            Parameter.dbParameter.values().toArray(new Parameter[]{})[row].setValue(d);
        }
        fireTableCellUpdated(row, col);
    }
}
