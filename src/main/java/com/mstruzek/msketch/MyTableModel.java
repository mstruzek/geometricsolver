package com.mstruzek.msketch;

import com.mstruzek.graphic.TableModelRemovable;

import javax.swing.table.AbstractTableModel;

public abstract class MyTableModel extends AbstractTableModel implements TableModelRemovable{

    private static final long serialVersionUID=1L;

    public MyTableModel(){
        super();
    }

}
