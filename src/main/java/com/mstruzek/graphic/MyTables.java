package com.mstruzek.graphic;

import com.mstruzek.controller.*;
import com.mstruzek.msketch.Constraint;
import com.mstruzek.msketch.GeometricObject;
import com.mstruzek.msketch.ModelRegistry;
import com.mstruzek.msketch.Parameter;

import javax.swing.*;
import javax.swing.event.MouseInputAdapter;
import javax.swing.event.TableModelEvent;
import javax.swing.event.TableModelListener;
import javax.swing.table.AbstractTableModel;
import javax.swing.table.TableColumn;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.util.HashSet;
import java.util.stream.Stream;

import static com.mstruzek.controller.EventType.CONSTRAINT_TABLE_INSERT;
import static com.mstruzek.controller.EventType.PARAMETER_TABLE_INSERT;
import static javax.swing.ScrollPaneConstants.HORIZONTAL_SCROLLBAR_NEVER;
import static javax.swing.ScrollPaneConstants.VERTICAL_SCROLLBAR_AS_NEEDED;

public class MyTables extends JPanel {

    private static final long serialVersionUID = 1L;

    private static final int D_WIDTH = 920;
    private static final int D_HEIGHT = 770;
    private static final int D_HEIGHT_JTAB = 250;

    /*** primitives */
    final AbstractTableModel ptm = new PrimitivesTableModel();
    /*** constraint */
    final AbstractTableModel mtm = new ConstraintsTableModel();
    /*** parameters */
    final AbstractTableModel vtm = new ParametersTableModel();

    final JTable constTable;
    final JTable primiTable;
    final JTable variaTable;

    final ContextMenuPopup popoupConstaint;
    final ContextMenuPopup popupPrimitives;

    public MyTables() {
        super();

        setPreferredSize(new Dimension(D_WIDTH, D_HEIGHT));
        //(new BorderLayout());

        ptm.addTableModelListener(new TableModelListener() {
            @Override
            public void tableChanged(TableModelEvent e) {
                Events.send(EventType.REFRESH_N_REPAINT, new Object[]{});
            }
        });

        constTable = new JTable(mtm);
        primiTable = new JTable(ptm);
        variaTable = new JTable(vtm);


        popoupConstaint = new ContextMenuPopup(e -> tableDeleteConstraint());
        popupPrimitives = new ContextMenuPopup(e -> tableDeletePrimitive());

        MouseInputAdapter mouseInputListener = new MouseInputAdapter() {
            @Override
            public void mouseReleased(MouseEvent e) {
                final JTable eventSourceTable = (JTable) e.getSource();
                if (eventSourceTable.equals(constTable)) {
                    popoupConstaint.show(e.getComponent(), e.getX(), e.getY());
                } else if (eventSourceTable.equals(primiTable)) {
                    popupPrimitives.show(e.getComponent(), e.getX(), e.getY());
                }
            }

        };
        constTable.addMouseListener(mouseInputListener);
        constTable.addMouseMotionListener(mouseInputListener);

        primiTable.addMouseListener(mouseInputListener);
        primiTable.addMouseMotionListener(mouseInputListener);

        setColumnsPreferredWidth();

        // Do the layout.
        JScrollPane primiScrollPane = new JScrollPane(primiTable, VERTICAL_SCROLLBAR_AS_NEEDED, HORIZONTAL_SCROLLBAR_NEVER);
        JScrollPane consScrollPane = new JScrollPane(constTable, VERTICAL_SCROLLBAR_AS_NEEDED, HORIZONTAL_SCROLLBAR_NEVER);
        JScrollPane variaScrollPane = new JScrollPane(variaTable, VERTICAL_SCROLLBAR_AS_NEEDED, HORIZONTAL_SCROLLBAR_NEVER);

        primiScrollPane.setPreferredSize(new Dimension(D_WIDTH, D_HEIGHT_JTAB));
        consScrollPane.setPreferredSize(new Dimension(D_WIDTH, D_HEIGHT_JTAB));
        variaScrollPane.setPreferredSize(new Dimension(D_WIDTH, D_HEIGHT_JTAB));

        add(primiScrollPane);
        add(consScrollPane);
        add(variaScrollPane);

        Events.addListener(EventType.REBUILD_TABLES, (eventType, arguments) -> {
            ptm.fireTableChanged(new TableModelEvent(ptm, 0, ptm.getRowCount(), TableModelEvent.ALL_COLUMNS, TableModelEvent.DELETE));
            vtm.fireTableChanged(new TableModelEvent(vtm, 0, vtm.getRowCount(), TableModelEvent.ALL_COLUMNS, TableModelEvent.DELETE));
            mtm.fireTableChanged(new TableModelEvent(mtm, 0, mtm.getRowCount(), TableModelEvent.ALL_COLUMNS, TableModelEvent.DELETE));
        });

        Events.addListener(EventType.PRIMITIVE_TABLE_INSERT, (eventType, arguments) -> {
            int rowCount = ptm.getRowCount();
            ptm.fireTableRowsInserted(rowCount, rowCount);
        });

        Events.addListener(CONSTRAINT_TABLE_INSERT, (eventType, arguments) -> {
            int rowCount = mtm.getRowCount();
            mtm.fireTableRowsInserted(rowCount, rowCount);
        });

        Events.addListener(PARAMETER_TABLE_INSERT, (eventType, arguments) -> {
            int rowCount = vtm.getRowCount();
            vtm.fireTableRowsInserted(rowCount, rowCount);
        });
    }

    private void setColumnsPreferredWidth() {
        TableColumn column = null;
        for (int i = 0; i < 8; i++) { //8
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
        for (int i = 0; i < 5; i++) {
            column = primiTable.getColumnModel().getColumn(i);
            switch (i) {
                case 0:
                    column.setPreferredWidth(10);
                    break;
                case 1:
                    column.setPreferredWidth(200);
                    break;
                case 2:
                    column.setPreferredWidth(10);
                    break;
                case 3:
                    column.setPreferredWidth(10);
                    break;
                case 4:
                    column.setPreferredWidth(10);
                    break;
            }
        }
    }

    private void tableDeleteConstraint() {
        final int i = constTable.getSelectionModel().getMinSelectionIndex();
        if (i < 0) return;
        Constraint constraint = ModelRegistry.dbConstraint.values()
                .stream()
                .filter(Constraint::isPersistent).skip(i)
                .findFirst()
                .orElseThrow(() -> new IndexOutOfBoundsException("constraint: " + i));

        int parId = constraint.getParameter();
        if (parId >= 0) {
            Parameter[] parameters = ModelRegistry.dbParameter().values().toArray(new Parameter[]{});
            for (int k = 0; k < parameters.length; k++) {
                if (parameters[k].getId() == parId) {
                    ModelRegistry.removeParameter(parId);
                    vtm.fireTableRowsDeleted(k, k);
                }
            }
        }
        ModelRegistry.removeConstraint(constraint);
        mtm.fireTableRowsDeleted(i, i);
    }

    private void tableDeletePrimitive() {

        final int i = primiTable.getSelectionModel().getMinSelectionIndex();
        if (i < 0) {
            return;
        }

        final GeometricObject geometric = ModelRegistry.dbPrimitives.values()
                .stream()
                .skip(i)
                .findFirst()
                .orElseThrow(() -> new IndexOutOfBoundsException("primitive : " + i));

        /*
         * przeszukaj wszystkie za aplikowane wiezy !
         */
        HashSet<Constraint> removable = new HashSet<>();
        for (Constraint c : ModelRegistry.dbConstraint.values()) {
            if (!c.isPersistent()) continue;
            boolean isBoundWith = Stream.of(geometric.getP1(),geometric.getP2(), geometric.getP3()).anyMatch(c::isBoundWith);
            if (isBoundWith) {
                removable.add(c);
            }
        }
        removable.forEach(ModelRegistry::removeConstraint);

        ModelRegistry.removeGeometric(geometric);

        Events.send(EventType.REBUILD_TABLES, null);
//        ptm.fireTableRowsDeleted(i, i);
    }


    /**
     * Klasa wenwetrzna odpowiedzialna za
     * widok podczas usuwania elementu z tabeli
     * wystarczy ze tabela implementuje interfejs
     * {@link TableModelRemovable}
     * @author root
     */
    class ContextMenuPopup extends JPopupMenu implements ActionListener {

        private final ActionListener delegate;

        ContextMenuPopup(ActionListener delegate) {
            super();
            this.delegate = delegate;
            JMenuItem bt = new JMenuItem("DELETE");
            bt.addActionListener(this);
            this.add(bt);
        }

        @Override
        public void actionPerformed(ActionEvent e) {
            delegate.actionPerformed(e);
        }
    }
}
