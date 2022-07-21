package com.mstruzek.controller;

import com.mstruzek.msketch.Constraint;
import com.mstruzek.msketch.ModelRegistry;

import javax.swing.table.AbstractTableModel;

public class ConstraintsTableModel extends AbstractTableModel {

    private static final String[] CONSTRAINT_COLUMN_NAMES = {"Id", "Type", "K", "L", "M", "N", "P-id", "Norm"};

    @Override
    public int getColumnCount() {
        return 8;
    }

    @Override
    public int getRowCount() {
        return (int) ModelRegistry.dbConstraint().values().stream().filter(Constraint::isPersistent).count();
    }

    @Override
    public Object getValueAt(int rowId, int colId) {
        int out;
        Constraint constraint =
            ModelRegistry.dbConstraint.values().stream().filter(Constraint::isPersistent).skip(rowId)
                .findFirst()
                .orElseThrow(() -> new IndexOutOfBoundsException("constraint: " + rowId));

        switch (colId) {
            case 0:
                return constraint.getConstraintId();
            case 1:
                return constraint.getConstraintType();
            case 2:
                return constraint.getK();
            case 3:
                out = constraint.getL();
                if (out == -1) return null;
                else return out;
            case 4:
                out = constraint.getM();
                if (out == -1) return null;
                else return out;
            case 5:
                out = constraint.getN();
                if (out == -1) return null;
                else return out;
            case 6:
                out = constraint.getParameter();
                if (out == -1) return null;
                else return out;
            case 7:
                return 0.0;
        }
        return null;
    }

    public String getColumnName(int col) {
        return CONSTRAINT_COLUMN_NAMES[col];
    }

    public Class getColumnClass(int c) {
        return String.class;
    }

    public boolean isCellEditable(int row, int col) {
        return false;
    }

}
