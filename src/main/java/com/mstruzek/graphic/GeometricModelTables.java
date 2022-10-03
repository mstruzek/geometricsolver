package com.mstruzek.graphic;

import com.mstruzek.controller.*;
import com.mstruzek.msketch.Constraint;
import com.mstruzek.msketch.GeometricObject;
import com.mstruzek.msketch.ModelRegistry;
import com.mstruzek.msketch.Parameter;

import javax.swing.*;
import javax.swing.event.TableModelEvent;
import javax.swing.event.TableModelListener;
import javax.swing.table.AbstractTableModel;
import java.awt.*;
import java.util.Arrays;
import java.util.HashSet;
import java.util.stream.Stream;

import static com.mstruzek.controller.EventType.CONSTRAINT_TABLE_INSERT;
import static com.mstruzek.controller.EventType.PARAMETER_TABLE_INSERT;
import static javax.swing.ScrollPaneConstants.HORIZONTAL_SCROLLBAR_NEVER;
import static javax.swing.ScrollPaneConstants.VERTICAL_SCROLLBAR_AS_NEEDED;

public class GeometricModelTables extends JPanel {

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

    public static final String DELETE_ACTION = "DELETE";
    final ContextActionSelector constraintActionSelector = new ContextActionSelector();
    final ContextActionSelector geometricActionSelector = new ContextActionSelector();

    public GeometricModelTables() {
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

        constraintActionSelector.registerAction(DELETE_ACTION, event -> tableDeleteConstraint());
        geometricActionSelector.registerAction(DELETE_ACTION, event -> tableDeletePrimitive());

        constTable.addMouseListener(constraintActionSelector);
        constTable.addMouseMotionListener(constraintActionSelector);

        primiTable.addMouseListener(geometricActionSelector);
        primiTable.addMouseMotionListener(geometricActionSelector);

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

        Events.addAwtListener(EventType.REBUILD_TABLES, (eventType, arguments) -> {
            ptm.fireTableChanged(new TableModelEvent(ptm, 0, ptm.getRowCount(), TableModelEvent.ALL_COLUMNS, TableModelEvent.DELETE));
            vtm.fireTableChanged(new TableModelEvent(vtm, 0, vtm.getRowCount(), TableModelEvent.ALL_COLUMNS, TableModelEvent.DELETE));
            mtm.fireTableChanged(new TableModelEvent(mtm, 0, mtm.getRowCount(), TableModelEvent.ALL_COLUMNS, TableModelEvent.DELETE));
        });

        Events.addAwtListener(EventType.PRIMITIVE_TABLE_INSERT, (eventType, arguments) -> {
            int rowCount = ptm.getRowCount();
            ptm.fireTableRowsInserted(rowCount, rowCount);

        });

        Events.addAwtListener(CONSTRAINT_TABLE_INSERT, (eventType, arguments) -> {
            int rowCount = mtm.getRowCount();
            mtm.fireTableRowsInserted(rowCount, rowCount);
        });

        Events.addAwtListener(PARAMETER_TABLE_INSERT, (eventType, arguments) -> {
            int rowCount = vtm.getRowCount();
            vtm.fireTableRowsInserted(rowCount, rowCount);
        });
    }

    private void setColumnsPreferredWidth() {
        //// constraints
//        constTable.getColumnModel().getColumn(0).setPreferredWidth(30);
        constTable.getColumnModel().getColumn(1).setPreferredWidth(120);
        constTable.getColumnModel().getColumn(2).setPreferredWidth(8);
        constTable.getColumnModel().getColumn(3).setPreferredWidth(8);
        constTable.getColumnModel().getColumn(4).setPreferredWidth(8);
        constTable.getColumnModel().getColumn(5).setPreferredWidth(8);
        constTable.getColumnModel().getColumn(6).setPreferredWidth(8);
        constTable.getColumnModel().getColumn(7).setPreferredWidth(60);

        //// geometric objects
        primiTable.getColumnModel().getColumn(0).setPreferredWidth(10);
        primiTable.getColumnModel().getColumn(1).setPreferredWidth(200);
        primiTable.getColumnModel().getColumn(2).setPreferredWidth(10);
        primiTable.getColumnModel().getColumn(3).setPreferredWidth(10);
        primiTable.getColumnModel().getColumn(4).setPreferredWidth(10);
    }

    private void tableDeleteConstraint() {
        int[] selectedRows = constTable.getSelectedRows();

        HashSet<Parameter> attachedParameters = new HashSet<>();
        HashSet<Constraint> selectedObjects = new HashSet<>();
        Arrays.stream(selectedRows).map(idx -> (Integer) constTable.getValueAt(idx, 0)).mapToObj(ModelRegistry.dbConstraint::get).forEach(constraint -> {
            int parameterId = constraint.getParameter();
            if (parameterId != -1) {
                attachedParameters.add(ModelRegistry.dbParameter.get(parameterId));
            }
            selectedObjects.add(constraint);

        });
        attachedParameters.forEach(ModelRegistry::removeParameter);
        selectedObjects.forEach(ModelRegistry::removeConstraint);

        Events.send(EventType.REBUILD_TABLES, null);
    }

    private void tableDeletePrimitive() {
        int[] selectedRows = primiTable.getSelectedRows();

        HashSet<Constraint> attachedCons = new HashSet<>();
        HashSet<GeometricObject> selectedObjects = new HashSet<>();
        Arrays.stream(selectedRows).map(idx -> (Integer) primiTable.getValueAt(idx, 0)).mapToObj(ModelRegistry.dbPrimitives::get).forEach(geometric -> {
            for (Constraint c : ModelRegistry.dbConstraint.values()) {
                if (!c.isPersistent()) continue;
                boolean isBoundWith = Stream.of(geometric.getP1(), geometric.getP2(), geometric.getP3()).anyMatch(c::isBoundWith);
                if (isBoundWith) {
                    attachedCons.add(c);
                }
            }
            selectedObjects.add(geometric);
        });
        attachedCons.forEach(ModelRegistry::removeConstraint);
        selectedObjects.forEach(ModelRegistry::removeGeometric);

        Events.send(EventType.REBUILD_TABLES, null);
//        ptm.fireTableRowsDeleted(i, i);
    }
}
