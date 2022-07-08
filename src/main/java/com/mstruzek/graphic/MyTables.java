package com.mstruzek.graphic;

import com.mstruzek.controller.*;
import com.mstruzek.msketch.GeometricPrimitive;

import javax.swing.*;
import javax.swing.event.MouseInputListener;
import javax.swing.event.TableModelEvent;
import javax.swing.event.TableModelListener;
import javax.swing.table.TableColumn;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;

public class MyTables extends JPanel implements MouseInputListener {

    /**
     *
     */
    private static final long serialVersionUID = 1L;

    private static final int D_WIDTH = 920;
    private static final int D_HEIGHT = 770;
    private static final int D_HEIGHT_JTAB = 250;


    /**
     * constraint
     */
    MyTableModel mtm = null;
    /**
     * primitives
     */
    MyTableModel ptm = null;
    /**
     * parameters
     */
    MyTableModel vtm = null;

    JTable constTable = null;
    JTable primiTable = null;
    JTable variaTable = null;

    ContextMenuPopup popoupConstaint;
    ContextMenuPopup popupPrimitives;

    // MPopup popupParameters = new MPopup(vtm,table3);

    public MyTables() {
        super();

        setPreferredSize(new Dimension(D_WIDTH, D_HEIGHT));
        //(new BorderLayout());
        this.mtm = new ConstraintsTableModel();
        this.ptm = new PrimitivesTableModel();
        this.vtm = new ParametersTableModel();

        ptm.addTableModelListener(new TableModelListener() {
            @Override
            public void tableChanged(TableModelEvent e) {
                EventBus.send(EventType.REFRESH_N_REPAINT, new Object[]{});
            }
        });

        constTable = new JTable(mtm);
        primiTable = new JTable(ptm);
        variaTable = new JTable(vtm);


        constTable.setPreferredScrollableViewportSize(new Dimension(D_WIDTH, D_HEIGHT_JTAB));
        primiTable.setPreferredScrollableViewportSize(new Dimension(D_WIDTH, D_HEIGHT_JTAB));
        variaTable.setPreferredScrollableViewportSize(new Dimension(D_WIDTH, D_HEIGHT_JTAB));

        //table.setFocusable(false);
        //table2.setFocusable(false);
        //table3.setFocusable(false);

        constTable.addMouseListener(this);
        constTable.addMouseMotionListener(this);

        primiTable.addMouseListener(this);
        primiTable.addMouseMotionListener(this);

        variaTable.addMouseListener(this);
        variaTable.addMouseMotionListener(this);

        TableColumn column = null;
        for(int i = 0; i < 1; i++) { //8
            column = constTable.getColumnModel().getColumn(i);
            switch (i) {
                case 0:
                    //column.setPreferredWidth(30);
                    break;
                case 1:
                    column.setPreferredWidth(120);
                    break;
                case 2:
                    column.setPreferredWidth(8);
                    break;
                case 3:
                    column.setPreferredWidth(8);
                    break;
                case 4:
                    column.setPreferredWidth(8);
                    break;
                case 5:
                    column.setPreferredWidth(8);
                    break;
                case 6:
                    column.setPreferredWidth(8);
                    break;
                case 7:
                    column.setPreferredWidth(60);
                    break;
            }
        }
        TableColumn column2 = null;
        for(int i = 0; i < 2; i++) {
            column2 = primiTable.getColumnModel().getColumn(i);
            switch (i) {
                case 0:
                    column2.setPreferredWidth(10);
                    break;
                case 1:
                    column2.setPreferredWidth(200);
                    break;
                case 2:
                    column2.setPreferredWidth(10);
                    break;
                case 3:
                    column2.setPreferredWidth(10);
                    break;
                case 4:
                    column2.setPreferredWidth(10);
                    break;
            }
        }

        //Do the layout.
        add(new JScrollPane(constTable));
        add(new JScrollPane(primiTable));
        add(new JScrollPane(variaTable));

        popoupConstaint = new ContextMenuPopup(mtm, constTable);
        popupPrimitives = new ContextMenuPopup(ptm, primiTable);

        EventBus.addListener(EventType.PRIMITIVE_TABLE_FIRE_INSERT, (eventType, arguments) -> {
            Integer firstRow = (Integer)arguments[0];
            Integer lastRow = (Integer)arguments[1];
            ptm.fireTableRowsInserted(firstRow, lastRow);
        });

        EventBus.addListener(EventType.CONSTRAINT_TABLE_FIRE_INSERT, (eventType, arguments) -> {
            Integer firstRow = (Integer)arguments[0];
            Integer lastRow = (Integer)arguments[1];
            mtm.fireTableRowsInserted(firstRow, lastRow);
        });

        EventBus.addListener(EventType.PARAMETER_TABLE_FIRE_INSERT, (eventType, arguments) -> {
            Integer firstRow = (Integer)arguments[0];
            Integer lastRow = (Integer)arguments[1];
            vtm.fireTableRowsInserted(firstRow, lastRow);
        });

        EventBus.addListener(EventType.PARAMETER_TABLE_FIRE_DELETE, (eventType, arguments) -> {
            Integer firstRow = (Integer)arguments[0];
            Integer lastRow = (Integer)arguments[1];
            vtm.fireTableRowsDeleted(firstRow, lastRow);
        });
    }

    @Override
    public void mouseReleased(MouseEvent e) {
        final JTable eventSourceTable = (JTable) e.getSource();
        if(eventSourceTable.equals(constTable)) {
            popoupConstaint.show(e.getComponent(), e.getX(), e.getY());
        } else if(eventSourceTable.equals(primiTable)) {
            popupPrimitives.show(e.getComponent(), e.getX(), e.getY());
        }
    }

    @Override
    public void mouseClicked(MouseEvent arg0) {
    }

    @Override
    public void mouseEntered(MouseEvent arg0) {
    }

    @Override
    public void mouseExited(MouseEvent arg0) {
    }

    @Override
    public void mousePressed(MouseEvent arg0) {
    }

    @Override
    public void mouseDragged(MouseEvent arg0) {
    }

    @Override
    public void mouseMoved(MouseEvent arg0) {
    }

    /**
     * Klasa wenwetrzna odpowiedzialna za
     * widok podczas usuwania elementu z tabeli
     * wystarczy ze tabela implementuje interfejs
     * {@link TableModelRemovable}
     *
     * @author root
     */
    class ContextMenuPopup extends JPopupMenu implements ActionListener {

        final TableModelRemovable model;
        final JTable table;

        ContextMenuPopup(TableModelRemovable model, JTable table) {
            super();
            this.model = model;
            this.table = table;

            JMenuItem bt = new JMenuItem("DELETE");
            bt.addActionListener(this);
            this.add(bt);
        }


        @Override
        public void actionPerformed(ActionEvent e) {
            model.remove(table.getSelectionModel().getMinSelectionIndex());
        }
    }
}
