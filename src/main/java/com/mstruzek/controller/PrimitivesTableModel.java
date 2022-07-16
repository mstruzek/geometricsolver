package com.mstruzek.controller;

import com.mstruzek.msketch.GeometricPrimitive;

import javax.swing.table.AbstractTableModel;

public class PrimitivesTableModel extends AbstractTableModel {

    private static final String[] PRIMITIVES_COLUMN_NAMES = {"id", "Type", "p1", "p2", "p3"};

    @Override
    public int getColumnCount() {
        return 5;
    }

    @Override
    public int getRowCount() {
        return GeometricPrimitive.dbPrimitives.size();
    }

    @Override
    public Object getValueAt(int rowId, int colmId) {
        int out;
        GeometricPrimitive primitive = GeometricPrimitive.dbPrimitives.values().toArray(new GeometricPrimitive[0])[rowId];
        switch (colmId) {
            case 0:
                return primitive.getPrimitiveId();
            case 1:
                return primitive.getType();
            case 2:
                return primitive.getP1();
            case 3:
                out = primitive.getP2();
                return (out == -1) ? null : out;
            case 4:
                out = primitive.getP3();
                return (out == -1) ? null : out;
        }
        return null;
    }

    public String getColumnName(int col) {
        return PRIMITIVES_COLUMN_NAMES[col];
    }

    public Class getColumnClass(int c) {
        return String.class;
    }

    public boolean isCellEditable(int row, int col) {
        return false;
    }

}
