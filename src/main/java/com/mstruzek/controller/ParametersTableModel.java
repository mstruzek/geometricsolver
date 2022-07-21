package com.mstruzek.controller;

import com.mstruzek.msketch.ModelRegistry;
import com.mstruzek.msketch.Parameter;

import javax.swing.table.AbstractTableModel;

public class ParametersTableModel extends AbstractTableModel {

    private static final String[] PARAMETERS_COLUMN_NAMES = {"id", "value"};

    @Override
    public int getColumnCount() {
        return 2;
    }

    @Override
    public int getRowCount() {
        return ModelRegistry.dbParameter().size();
    }

    @Override
    public Object getValueAt(int rowIndex, int columnIndex) {
        Parameter parameter = ModelRegistry.dbParameter().values().toArray(new Parameter[0])[rowIndex];
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
        if (col > 0) return true;
        return false;
    }

    public void setValueAt(Object value, int row, int col) {
        double parameterValue = Double.parseDouble(value.toString());
        if (col > 0) {
            Parameter parameter = ModelRegistry.dbParameter().values().toArray(new Parameter[]{})[row];
            ModelRegistry.setParameterValue(parameter.getId(), parameterValue);
        }
        fireTableCellUpdated(row, col);
    }
}
