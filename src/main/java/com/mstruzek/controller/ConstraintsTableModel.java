package com.mstruzek.controller;

import com.mstruzek.msketch.*;

public class ConstraintsTableModel extends MyTableModel{

    private static final String[] CONSTRAINT_COLUMN_NAMES={"Id","Type","K","L","M","N","P-id","Norm"};

    public void remove(int i){
        if(i<0) return;
        int id=Model.constraintContainer.get(i).getConstraintId();
        int parId=Model.constraintContainer.get(i).getParametr();
        if(parId>=0){
            for(int k=0;k<Model.parameterContainer.size();k++){
                if(Model.parameterContainer.get(k).getId()==parId){
                    Model.parameterContainer.remove(k);
                    Parameter.dbParameter.remove(parId);
                    EventBus.send(EventType.PARAMETER_TABLE_FIRE_DELETE,new Object[]{k,k});
                }
            }
        }
        Model.constraintContainer.remove(i);
        Constraint.dbConstraint.remove(id);
        fireTableRowsDeleted(i,i);

    }

    @Override
    public int getColumnCount(){
        return 7;
    }

    @Override
    public int getRowCount(){
        return Model.constraintContainer.size();
    }

    @Override
    public Object getValueAt(int wiersz,int kolumna){
        int out;
        switch(kolumna){
            case 0:
                //return "id";
                return Model.constraintContainer.get(wiersz).getConstraintId();
//                        return constraintContainer.get(wiersz).getConstraintType();
            case 1:
                //return "type";
                return Model.constraintContainer.get(wiersz).getConstraintType();
            case 2:
                //return "K";
                return Model.constraintContainer.get(wiersz).getK();
            case 3:
                //return "L";
                out=Model.constraintContainer.get(wiersz).getL();
                if(out==-1) return null;
                else return out;
            case 4:
                //return "M";
                out=Model.constraintContainer.get(wiersz).getM();
                if(out==-1) return null;
                else return out;
            case 5:
                //return "N";
                out=Model.constraintContainer.get(wiersz).getN();
                if(out==-1) return null;
                else return out;
            case 6:
                //return "P-id";
                out=Model.constraintContainer.get(wiersz).getParametr();
                if(out==-1) return null;
                else return out;
            case 7:
                return Model.constraintContainer.get(wiersz).getNorm(Point.dbPoint,Parameter.dbParameter);
        }
        return null;
    }

    public String getColumnName(int col){
        return CONSTRAINT_COLUMN_NAMES[col];
    }

    public Class getColumnClass(int c){
        //return getValueAt(0, c).getClass();
        return String.class;
    }

    public boolean isCellEditable(int row,int col){
        return false;
    }

}
