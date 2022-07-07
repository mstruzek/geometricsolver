package com.mstruzek.controller;

import com.mstruzek.msketch.*;

public class PrimitivesTableModel extends MyTableModel{

    private static final String[] PRIMITIVES_COLUMN_NAMES={"id","Type","p1","p2","p3"};

    @Override
    public void remove(int i){
        if(i<0) return;
        int id=Model.primitivesContainer.get(i).getPrimitiveId();

        //usun wiezy powiazane
        int[] con=Model.primitivesContainer.get(i).associateConstraintsId();
        for(int k=0;k<con.length;k++){
            Constraint.dbConstraint.remove(con[k]);
        }
        //usun punkty
        for(int pi: Model.primitivesContainer.get(i).getAllPointsId()){
            Point.dbPoint.remove(pi);
        }

        Model.primitivesContainer.remove(i);
        GeometricPrimitive.dbPrimitives.remove(id);
        fireTableRowsDeleted(i,i);

        //FIXME - JESZCE PUNKTY TRZEBA USUNAC
        System.out.println(GeometricPrimitive.dbPrimitives);
        System.out.println(Constraint.dbConstraint);
    }

    @Override
    public int getColumnCount(){
        return 4;
    }

    @Override
    public int getRowCount(){
        return Model.primitivesContainer.size();
    }

    @Override
    public Object getValueAt(int rowId,int colmId){
        int out;
        switch(colmId){
            case 0:
                return Model.primitivesContainer.get(rowId).getPrimitiveId();
            case 1:
                return Model.primitivesContainer.get(rowId).getType();
            case 2:
                return Model.primitivesContainer.get(rowId).getP1();
            case 3:
                out=Model.primitivesContainer.get(rowId).getP2();
                return (out==-1) ? null:out;
            case 4:
                out=Model.primitivesContainer.get(rowId).getP3();
                return (out==-1) ? null:out;
        }
        return null;
    }

    public String getColumnName(int col){
        return PRIMITIVES_COLUMN_NAMES[col];
    }

    public Class getColumnClass(int c){
        return String.class;
    }

    public boolean isCellEditable(int row,int col){
        return false;
    }

}
