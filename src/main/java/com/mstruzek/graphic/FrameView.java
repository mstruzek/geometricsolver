package com.mstruzek.graphic;

import com.mstruzek.controller.Controller;
import com.mstruzek.controller.EventType;
import com.mstruzek.controller.Events;
import com.mstruzek.msketch.ConstraintType;
import com.mstruzek.msketch.ModelRegistry;
import com.mstruzek.msketch.solver.GeometricSolverType;
import com.mstruzek.utils.Dispatcher;
import com.mstruzek.utils.Dispatchers;

import javax.swing.*;
import javax.swing.GroupLayout.Alignment;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.*;
import java.awt.event.*;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.time.Instant;
import java.util.Objects;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

import static com.mstruzek.graphic.Property.*;
import static javax.swing.SwingConstants.VERTICAL;

public class FrameView extends JFrame {

    public static final int L_WIDTH = 1620;
    public static final int R_HEIGHT = 1400;
    public static final int L_HEIGHT = 1400;
    public static final int R_WIDTH = 1420;

    public static final int CONSOLE_WIDTH = 920;
    public static final int CONSOLE_OUTPUT_HEIGHT = 420;

    public static final Color ON_CPU_COLOR = new Color(0, 181, 245);
    public static final Color ON_GPU_COLOR = new Color(118, 185, 0);
    public static final Color CONSTRAINT_BORDER_COLOR = Color.DARK_GRAY;
    public static final Color CONSTRAINT_PANEL_BACKGROUND_COLOR = new Color(244, 249, 192);
    public static final Color HELP_PANEL_BACKGROUND_COLOR = new Color(100, 255, 100, 50);
    public static final Color SKETCH_INFO_BORDER_COLOR = Color.darkGray;
    public static final Color SKETCH_INFO_BACKGROUND_COLOR = new Color(131, 188, 252);
    public static final Color FRAME_BACKGROUND_COLOR = null; /// Default Wash
    public static final String FRAME_TITLE = "GCS GeometricSolver 2009-2022";
    public static final int SOLVER_PANEL_HEIGHT = 140;
    public static final int SOLVER_PANEL_WIDTH = 920;

    ///  Toolbar Actions
    public static final String COMMAND_LOAD = "Load";
    public static final String COMMAND_STORE = "Store";
    public static final String COMMAND_CLEAR = "Clear";
    public static final String COMMAND_NORMAL1 = "Normal";
    public static final String COMMAND_DRAW_LINE = "Draw Line";
    public static final String COMMAND_DRAW_CIRCLE = "Draw Circle";
    public static final String COMMAND_DRAW_ARC = "Draw Arc";
    public static final String COMMAND_DRAW_POINT = "Draw Point";
    public static final String COMMAND_REFRESH = "REFRESH";
    public static final String COMMAND_SOLVE = "SOLVE";
    public static final String COMMAND_REPOZ = "Repoz";
    public static final String COMMAND_RELAX = "Relax";
    public static final String COMMAND_CTRL = "CTRL";
    public static final String DEFAULT_LOAD_DIRECTORY = "e:\\source\\gsketcher\\data\\";

    /// acceptable model extension
    public static final String FILE_EXTENSION_GCM = "gcm";

    private Container pane = getContentPane();

    /**
     * zmienna na parametry
     */
    final JTextField parameterField = new JTextField(5);

    {
        parameterField.setText("10.0");
    }

    /**
     * tabelki z wiezami,figurami i parametrami
     */
    final MyTables myTables;

    /**
     * wypisujemy tutaj wszystko co idzie na System.out.println();
     */
    final JTextArea consoleOutput = new JTextArea();

    final JScrollPane consoleScrollPane;

    /**
     * glowny controller
     */
    final Controller controller;

    /// widok na pojemnik K,L,M,N
    final JLabel klmn = new JLabel("K,L,M,N", SwingConstants.CENTER);

    final JLabel pickedPoint = new JLabel("", SwingConstants.CENTER);

    /// wyswietla aktualna pozycje kursora
    final JLabel currentPosition = new JLabel("Currrent Position:");

    final MySketch ms;

    /// Single ordered executor
    final Dispatcher controllerEventQueue = Dispatchers.newDispatcher();

    public FrameView(Controller controller) {
        super(FRAME_TITLE);
        this.controller = controller;

        pane.setLayout(new BorderLayout());

        setFocusCycleRoot(true);
        setFocusable(true);

        addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                FrameView.this.requestFocusInWindow();
                super.mouseClicked(e);
            }
        });

        /// Rozklad calosiowy
        JPanel main = new JPanel();
        main.setLayout(new BoxLayout(main, BoxLayout.X_AXIS));

        /// -----------------------------------------------------------------
        JPanel left = new JPanel();
        JPanel right = new JPanel();

        left.setPreferredSize(new Dimension(L_WIDTH, L_HEIGHT));
        right.setPreferredSize(new Dimension(R_WIDTH, R_HEIGHT));

        left.setBackground(FRAME_BACKGROUND_COLOR);
        right.setBackground(FRAME_BACKGROUND_COLOR);

        /// left.setBorder(BorderFactory.createLineBorder(Color.CYAN));
        /// right.setBorder(BorderFactory.createLineBorder(Color.CYAN));
        main.add(left);
        main.add(right);

        /// -----------------------------------------------------------------
        // SZKICOWNIK
        ms = new MySketch(this.controller);
        ms.setControlGuidelines(false);
        ms.addPropertyChangeListener(new PropertyChangeListener() {
            @Override
            public void propertyChange(PropertyChangeEvent evt) {
                switch (evt.getPropertyName()) {
                    case SELECTED_POINT:
                        pickedPoint.setText((String) evt.getNewValue());
                    case KLMN_POINTS:
                        klmn.setText((String) evt.getNewValue());
                        return;
                    case CURRENT_POSITION:
                        currentPosition.setText((String) evt.getNewValue());
                        return;
                    default:
                }
            }
        });
        left.add(ms);

        /// -----------------------------------------------------------------
        JPanel panPoints = setupSketchInfoPanel();
        left.add(panPoints);

        JPanel constraintPanel = setupConstraintPanel(controller);
        left.add(constraintPanel);

        /// -----------------------------------------------------------------
        // Tabelki
        myTables = new MyTables();
        myTables.setFocusable(false);

        right.add(myTables);

        // additional worker for aggregated log publication on swing thread
        consoleDownstream.submit(this::consoleObservable);

        consoleOutput.setFont(new Font("Courier New", Font.PLAIN, 12));
        consoleScrollPane = new JScrollPane(consoleOutput);
        consoleScrollPane.setPreferredSize(new Dimension(CONSOLE_WIDTH, CONSOLE_OUTPUT_HEIGHT));
        consoleScrollPane.scrollRectToVisible(consoleOutput.getVisibleRect());
        redirectStdErrOut();

        /// -----------------------------------------------------------------
        SolverStatPanel solverStatPanel = new SolverStatPanel();
        solverStatPanel.setPreferredSize(new Dimension(SOLVER_PANEL_WIDTH, SOLVER_PANEL_HEIGHT));
        right.add(solverStatPanel);
        right.add(consoleScrollPane);

        /// -----------------------------------------------------------------
        Events.addListener(EventType.REFRESH_N_REPAINT, (eventType, arguments) -> {
            ms.rebuildViewModel();
            ms.repaintLater();
        });

        Events.addAwtListener(EventType.CONTROLLER_ERROR, (eventType, arguments) -> {
            JOptionPane.showMessageDialog(this, (String) arguments[0], "Application Error", JOptionPane.ERROR_MESSAGE);
        });

        /// -----------------------------------------------------------------
        /// Toolbar i srodkowe okno
        JToolBar actionToolbar = setupActionToolBar(controller);

        JToolBar constraintToolbar = setupConstraintToolBar(controller);

        /// -----------------------------------------------------------------
        ///
        pane.add(actionToolbar, BorderLayout.NORTH);
        pane.add(constraintToolbar, BorderLayout.WEST);
        pane.add(main, BorderLayout.CENTER);
    }


    private JPanel setupSketchInfoPanel() {
        JPanel panPoints = new JPanel();
        panPoints.setLayout(new BorderLayout());
        panPoints.setBackground(SKETCH_INFO_BACKGROUND_COLOR);
        panPoints.setPreferredSize(new Dimension(SOLVER_PANEL_WIDTH, 60));
        panPoints.setBorder(BorderFactory.createLineBorder(SKETCH_INFO_BORDER_COLOR));
        panPoints.add(klmn, BorderLayout.NORTH);
        panPoints.add(currentPosition, BorderLayout.SOUTH);
        panPoints.add(pickedPoint, BorderLayout.EAST);
        return panPoints;
    }

    private JPanel setupConstraintPanel(Controller controller) {
        /// Dodawanie wiezow
        JPanel constraintPanel = new JPanel();
        GroupLayout groupLayout = new GroupLayout(constraintPanel);
        constraintPanel.setLayout(groupLayout);
        constraintPanel.setBackground(CONSTRAINT_PANEL_BACKGROUND_COLOR);
        constraintPanel.setPreferredSize(new Dimension(680, 250));
        constraintPanel.setBorder(BorderFactory.createTitledBorder(BorderFactory.createLineBorder(CONSTRAINT_BORDER_COLOR), "Add Constraint"));

        final JTextArea consDescr = new JTextArea(7, 40);
        consDescr.setBorder(BorderFactory.createTitledBorder(BorderFactory.createEmptyBorder(30, 20, 20, 20), "HELP"));
        consDescr.setLineWrap(true);
        consDescr.setWrapStyleWord(true);
        consDescr.setEditable(false);
        consDescr.setFocusable(false);
        consDescr.setBackground(HELP_PANEL_BACKGROUND_COLOR);

        final JComboBox<ConstraintType> combo = new JComboBox<>(ConstraintType.values());
        combo.setFocusable(false);
        combo.setPreferredSize(new Dimension(240, -1));
        combo.addActionListener(actionEvent -> {
            JComboBox<ConstraintType> cb = (JComboBox<ConstraintType>) actionEvent.getSource();
            ConstraintType what = (ConstraintType) cb.getSelectedItem();
            consDescr.setText(Objects.requireNonNull(what).getHelp());
            if (consDescr.getParent() != null) {
                consDescr.getParent().repaint();
            }
        });

        combo.setSelectedItem(ConstraintType.FixPoint);

        AbstractAction action = new AbstractAction("Add Constraint", null) {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                controllerEventQueue.submit(() -> {
                    ConstraintType constraintType = (ConstraintType) combo.getSelectedItem();

                    controller.addConstraint(constraintType,
                        ms.getPointK(),
                        ms.getPointL(),
                        ms.getPointM(),
                        ms.getPointN(),
                        Double.parseDouble(parameterField.getText())
                    );
                    ms.clearAll();
                    requestFocus();
                });
            }
        };
        action.putValue(Action.SHORT_DESCRIPTION, "Add constraint into model");
        action.putValue(Action.MNEMONIC_KEY, KeyEvent.VK_C);
        JButton constraintButton = new JButton(action);

        /// -----------------------------------------------------------------
        /*
         * Constraint View default layout
         */
        groupLayout.setAutoCreateGaps(true);
        groupLayout.setHorizontalGroup(groupLayout
            .createSequentialGroup().addGroup(groupLayout.createParallelGroup(Alignment.LEADING)
                .addComponent(combo)
                .addComponent(parameterField))
            .addGroup(groupLayout.createParallelGroup(Alignment.LEADING)
                .addComponent(consDescr)
                .addComponent(constraintButton)
            )
        );

        groupLayout.setVerticalGroup(groupLayout
            .createSequentialGroup().addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
                .addComponent(combo)
                .addComponent(consDescr))
            .addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
                .addComponent(parameterField)
                .addComponent(constraintButton))
        );
        return constraintPanel;
    }

    private JToolBar setupActionToolBar(Controller controller) {
        // ToolBar
        JToolBar jToolBar = new JToolBar();
        /// -----------------------------------------------------------------
        final ActionListener loadActionListener = actionEvent -> {
            JFileChooser jFileChooser = new JFileChooser();
            jFileChooser.setCurrentDirectory(new File(DEFAULT_LOAD_DIRECTORY));
            jFileChooser.setFileFilter(new FileNameExtensionFilter("Geometric Constraint Model File", FILE_EXTENSION_GCM));
            int response = jFileChooser.showOpenDialog(null);
            if (response == JFileChooser.APPROVE_OPTION) {
                controllerEventQueue.submit(() -> controller.readModelFrom(jFileChooser.getSelectedFile()));
            }
        };
        /// -----------------------------------------------------------------
        final ActionListener solveActionListener = actionEvent -> controllerEventQueue.submit(() -> {
            controller.solveSystem();
            ms.rebuildViewModel();
            ms.repaintLater();
        });

        /// -----------------------------------------------------------------
        final ActionListener repositionActionListener = actionEvent -> controllerEventQueue.submit(() -> {
            controller.evaluateGuidePoints();
            ms.rebuildViewModel();
            ms.repaintLater();
        });

        /// -----------------------------------------------------------------
        final ActionListener relaxeActionListener = actionEvent -> controllerEventQueue.submit(() -> {
            controller.relaxControlPoints();
            controller.evaluateGuidePoints();
            ms.rebuildViewModel();
            ms.repaintLater();
        });

        /// -----------------------------------------------------------------
        final ActionListener storeActionListener = actionEvent -> {
            JFileChooser jFileChooser = new JFileChooser();
            jFileChooser.setCurrentDirectory(new File(DEFAULT_LOAD_DIRECTORY));
            jFileChooser.setFileFilter(new FileNameExtensionFilter("Geometric Constraint Model File", FILE_EXTENSION_GCM));
            int response = jFileChooser.showSaveDialog(null);
            if (response == JFileChooser.APPROVE_OPTION) {
                controllerEventQueue.submit(() -> controller.writeModelInto(jFileChooser.getSelectedFile()));
            }
        };

        /// -----------------------------------------------------------------
        final ActionListener clearActionListener = actionEvent -> controllerEventQueue.submit(() -> {
            ModelRegistry.removeObjectsFromModel();
            ms.rebuildViewModel();
            ms.repaintLater();
            clearTextArea();
            Events.send(EventType.REBUILD_TABLES, null);
        });

        /// -----------------------------------------------------------------
        final ActionListener refreshActionListener = actionEvent -> {
            ms.rebuildViewModel();
            ms.repaintLater();
        };
        final ActionListener cntrlActionListener = actionEvent -> {
            ms.setControlGuidelines(!ms.isControlGuidelines());
            ms.repaintLater();
        };
        /// -----------------------------------------------------------------
        final ActionListener sketchStateNormalListener = event -> ms.setState(MySketchState.Normal);
        final ActionListener drawLineActionListener = event -> ms.setState(MySketchState.DrawLine);
        final ActionListener drawCircleActionListener = event -> ms.setState(MySketchState.DrawCircle);
        final ActionListener drawArcActionListener = event -> ms.setState(MySketchState.DrawArc);
        final ActionListener drawPointActionListener = event -> ms.setState(MySketchState.DrawPoint);

        /// -----------------------------------------------------------------
        final ActionListener solverActionListener = actionEvent -> controllerEventQueue.submit(() -> {
            final String actionCommand = actionEvent.getActionCommand();
            controller.setSolverType(GeometricSolverType.valueOf(actionCommand));
        });

        /// -----------------------------------------------------------------
        final ButtonGroup solverType = new ButtonGroup();
        final JRadioButton onCPU = new JRadioButton("CPU colt", true);
        final JRadioButton onGPU = new JRadioButton("GPU cuSolver", true);
        onCPU.setActionCommand(GeometricSolverType.CPU_SOLVER.name());
        onGPU.setActionCommand(GeometricSolverType.GPU_SOLVER.name());
        onCPU.setBackground(ON_CPU_COLOR);
        onGPU.setBackground(ON_GPU_COLOR);
        onCPU.setFocusable(false);
        onGPU.setFocusable(false);
        solverType.add(onCPU);
        solverType.add(onGPU);
        onCPU.addActionListener(solverActionListener);
        onGPU.addActionListener(solverActionListener);
        /// -----------------------------------------------------------------

        class ComponentBuilder {
            public JButton newButton(String title, Integer mnemonic, Color backgroundColor, ActionListener actionListener) {
                JButton button = new JButton(title);
                button.setFocusable(false);
                if (mnemonic != null)
                    button.setMnemonic(mnemonic);
                if (backgroundColor != null)
                    button.setBackground(backgroundColor);
                if (actionListener != null)
                    button.addActionListener(actionListener);
                return button;
            }
        }
        /// -----------------------------------------------------------------
        ComponentBuilder builder = new ComponentBuilder();
        jToolBar.addSeparator(new Dimension(20, 1));
        jToolBar.add(builder.newButton(COMMAND_LOAD, KeyEvent.VK_D, null, loadActionListener));
        jToolBar.add(builder.newButton(COMMAND_STORE, null, null, storeActionListener));
        jToolBar.add(builder.newButton(COMMAND_CLEAR, null, null, clearActionListener));
        jToolBar.addSeparator(new Dimension(20, 1));
        jToolBar.add(builder.newButton(COMMAND_NORMAL1, null, null, sketchStateNormalListener));
        jToolBar.add(builder.newButton(COMMAND_DRAW_LINE, null, null, drawLineActionListener));
        jToolBar.add(builder.newButton(COMMAND_DRAW_CIRCLE, null, null, drawCircleActionListener));
        jToolBar.add(builder.newButton(COMMAND_DRAW_ARC, null, null, drawArcActionListener));
        jToolBar.add(builder.newButton(COMMAND_DRAW_POINT, null, null, drawPointActionListener));
        jToolBar.addSeparator(new Dimension(20, 1));
        jToolBar.add(builder.newButton(COMMAND_REFRESH, null, null, refreshActionListener));
        jToolBar.add(builder.newButton(COMMAND_SOLVE, KeyEvent.VK_S, Color.GREEN, solveActionListener));
        jToolBar.addSeparator(new Dimension(20, 1));
        jToolBar.add(builder.newButton(COMMAND_REPOZ, KeyEvent.VK_Z, null, repositionActionListener));
        jToolBar.add(builder.newButton(COMMAND_RELAX, KeyEvent.VK_X, null, relaxeActionListener));
        jToolBar.add(builder.newButton(COMMAND_CTRL, null, Color.CYAN, cntrlActionListener));
        jToolBar.addSeparator(new Dimension(120, 1));
        jToolBar.add(onCPU);
        jToolBar.addSeparator(new Dimension(20, 1));
        jToolBar.add(onGPU);
        return jToolBar;
    }

    private JToolBar setupConstraintToolBar(Controller controller) {
        JToolBar toolbar = new JToolBar();
        //toolbar.setBorder(BorderFactory.createLineBorder(Color.green, 1));
        toolbar.setLayout(new BoxLayout(toolbar, BoxLayout.Y_AXIS));
        toolbar.setAlignmentX(Component.CENTER_ALIGNMENT);
        ActionListener constraintsActionListener = actionEvent -> {
            ConstraintType constraintType = ConstraintType.valueOf(actionEvent.getActionCommand());
            controllerEventQueue.submit(() -> {
                controller.addConstraint(constraintType,
                    ms.getPointK(),
                    ms.getPointL(),
                    ms.getPointM(),
                    ms.getPointN(),
                    Double.parseDouble(parameterField.getText())
                );
                ms.clearAll();
                requestFocus();
            });
        };
        class ComponentBuilder {
            static final int WIDTH = 30;
            static final int HEIGHT = 30;
            public JButton newButton(String title, int mnemonic, String actionCommand) {
                JButton button = new JButton(title);
                button.setMnemonic(mnemonic);
                button.setActionCommand(actionCommand);
                button.setToolTipText(actionCommand);
                button.addActionListener(constraintsActionListener);
                button.setAlignmentX(Component.CENTER_ALIGNMENT);
                button.setMinimumSize(new Dimension(WIDTH, HEIGHT));
                button.setSize(new Dimension(WIDTH, HEIGHT));
                button.setMaximumSize(new Dimension(WIDTH, HEIGHT));
                return button;
            }
        }
        ComponentBuilder builder = new ComponentBuilder();
        toolbar.add(builder.newButton("H", KeyEvent.VK_H, ConstraintType.SetHorizontal.name()));
        toolbar.add(builder.newButton("V", KeyEvent.VK_V, ConstraintType.SetVertical.name()));
        toolbar.add(builder.newButton("L", KeyEvent.VK_L, ConstraintType.LinesPerpendicular.name()));
        toolbar.add(builder.newButton("LP", KeyEvent.VK_P, ConstraintType.LinesParallelism.name()));
        toolbar.add(builder.newButton("C2", KeyEvent.VK_2, ConstraintType.Connect2Points.name()));
        toolbar.add(builder.newButton("TG", KeyEvent.VK_T, ConstraintType.Tangency.name()));
        toolbar.add(builder.newButton("Eq", KeyEvent.VK_Q, ConstraintType.EqualLength.name()));
        toolbar.add(builder.newButton("A", KeyEvent.VK_A, ConstraintType.Angle2Lines.name()));
        toolbar.add(builder.newButton("OT", KeyEvent.VK_O, ConstraintType.CircleTangency.name()));
        toolbar.add(builder.newButton("hP", KeyEvent.VK_7, ConstraintType.HorizontalPoint.name()));
        toolbar.add(builder.newButton("vP", KeyEvent.VK_8, ConstraintType.VerticalPoint.name()));
        toolbar.add(builder.newButton("xP", KeyEvent.VK_4, ConstraintType.FixPoint.name()));
        toolbar.setOrientation(VERTICAL);
        return toolbar;
    }

    private void clearTextArea() {
        SwingUtilities.invokeLater(() -> {
            consoleOutput.setText("");
            consoleOutput.setCaretPosition(consoleOutput.getDocument().getLength()); /// auto scroll - follow caret position
        });
    }

    private static final BlockingQueue<String> messageQueue = new LinkedBlockingQueue<>(64);
    private static final Dispatcher consoleDownstream = Dispatchers.newDispatcher();
    private static final long AWAIT_BUFFER_TIMEOUT = 150;   // ms
    public static final int MQ_POLL_TIMEOUT = 70;          // ms

    /**
     * Stdout into swing TextArea console view
     * @param text
     */
    private void updateTextArea(final String text) {
        try {
            messageQueue.put(text);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    private void consoleObservable() {
        long batchStart = 0;
        String text = null;
        StringBuffer buffer = new StringBuffer();
        while (true) {
            try {
                text = messageQueue.poll(MQ_POLL_TIMEOUT, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException(e);
            }
            if (text != null) {
                if (buffer.isEmpty()) {
                    batchStart = Instant.now().toEpochMilli();
                }
                buffer.append(text);
            }
            long elapsed = Instant.now().toEpochMilli() - batchStart;
            if (elapsed > AWAIT_BUFFER_TIMEOUT) {
                final String send = buffer.toString();
                buffer = new StringBuffer();
                SwingUtilities.invokeLater(() -> {
                    consoleOutput.append(send);
                    consoleOutput.setCaretPosition(consoleOutput.getDocument().getLength()); /// auto scroll - follow caret position
                });
            }
        }
    }

    private void redirectStdErrOut() {
        System.setOut(new PrintStream(new OutputStreamDelegate(System.out), true));
        System.setErr(new PrintStream(new OutputStreamDelegate(System.err), true));
    }

    private class OutputStreamDelegate extends java.io.OutputStream {

        private final java.io.OutputStream parentStream;

        public OutputStreamDelegate(java.io.OutputStream parentStream) {
            this.parentStream = parentStream;
        }

        @Override
        public void write(int b) throws IOException {
            parentStream.write(b);
            updateTextArea(String.valueOf((char) b));
        }

        @Override
        public void write(byte[] b, int off, int len) throws IOException {
            parentStream.write(b, off, len);
            updateTextArea(new String(b, off, len));
        }

        @Override
        public void write(byte[] b) throws IOException {
            parentStream.write(b);
            updateTextArea(new String(b));
        }
    }
}
