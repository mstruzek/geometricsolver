package com.mstruzek.graphic;

import com.mstruzek.controller.ActiveKey;
import com.mstruzek.controller.Controller;
import com.mstruzek.controller.EventType;
import com.mstruzek.controller.Events;
import com.mstruzek.msketch.ConstraintType;
import com.mstruzek.msketch.ModelRegistry;
import com.mstruzek.msketch.solver.GeometricSolverType;

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
import java.net.URL;
import java.util.Objects;
import java.util.concurrent.*;

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
    public static final String COMMAND_CPU = "CPU";
    public static final String COMMAND_GPU = "GPU";
    public static final String DEFAULT_LOAD_DIRECTORY = "e:\\source\\gsketcher\\data\\";

    /// acceptable model extension
    public static final String FILE_EXTENSION_GCM = "gcs";
    public static final String COMMAND_LOAD_MODEL_DESCRIPTION = "Load  model from ... ";
    public static final String COMMAND_SOLVE_DESCRIPTION = "Run selected solver";
    public static final String COMMAND_REPOZ_DESCRIPTION = "Reposition constraint";
    public static final String COMMAND_RELAX_DESCRIPTION = "Relax geometric points and reposition !";


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
    final ExecutorService controllerEventQueue = Executors.newSingleThreadExecutor();

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
        addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                switch (e.getKeyChar()) {
                    case 'k':
                        ms.setAck(ActiveKey.K);
                        break;
                    case 'l':
                        ms.setAck(ActiveKey.L);
                        break;
                    case 'm':
                        ms.setAck(ActiveKey.M);
                        break;
                    case 'n':
                        ms.setAck(ActiveKey.N);
                        break;
                    default:
                        ms.setAck(ActiveKey.None);
                        break;
                }
                super.keyPressed(e);
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

        consoleOutput.setFont(new Font("Courier New", Font.PLAIN, 12));
        consoleScrollPane = new JScrollPane(consoleOutput);
        consoleScrollPane.setPreferredSize(new Dimension(CONSOLE_WIDTH, CONSOLE_OUTPUT_HEIGHT));
        consoleScrollPane.scrollRectToVisible(consoleOutput.getVisibleRect());
        //redirectStdErrOut();

        /// -----------------------------------------------------------------
        SolverStatPanel solverStatPanel = new SolverStatPanel();
        solverStatPanel.setPreferredSize(new Dimension(SOLVER_PANEL_WIDTH, SOLVER_PANEL_HEIGHT));
        right.add(solverStatPanel);
        right.add(consoleScrollPane);

        /// -----------------------------------------------------------------
        /// Toolbar i srodkowe okno
        JToolBar actionToolbar = setupdActionToolBar(controller);

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

        final JComboBox combo = new JComboBox(ConstraintType.values());
        combo.setFocusable(false);
        combo.setPreferredSize(new Dimension(240, -1));
        combo.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                JComboBox cb = (JComboBox) e.getSource();
                ConstraintType what = (ConstraintType) cb.getSelectedItem();
                consDescr.setText(Objects.requireNonNull(what).getHelp());
                if (consDescr.getParent() != null) {
                    consDescr.getParent().repaint();
                }
            }
        });

        combo.setSelectedItem(ConstraintType.FixPoint);

        JButton constraintButton = ActionBuilder.create("Add Constraint", null, "Add constraint into model", KeyEvent.VK_C, new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
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
            }
        }, controllerEventQueue);

        /// -----------------------------------------------------------------
        /*
         * Constraint View default layout
         */
        groupLayout.setAutoCreateGaps(true);
        groupLayout.setHorizontalGroup(
            groupLayout.createSequentialGroup()
                .addGroup(groupLayout.createParallelGroup(Alignment.LEADING)
                    .addComponent(combo)
                    .addComponent(parameterField))
                .addGroup(groupLayout.createParallelGroup(Alignment.LEADING)
                    .addComponent(consDescr)
                    .addComponent(constraintButton)
                )
        );

        groupLayout.setVerticalGroup(
            groupLayout.createSequentialGroup()
                .addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
                    .addComponent(combo)
                    .addComponent(consDescr))
                .addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
                    .addComponent(parameterField)
                    .addComponent(constraintButton))
        );
        return constraintPanel;
    }



    private JToolBar setupdActionToolBar(Controller controller) {
        // ToolBar
        JToolBar jToolBar = new JToolBar();
        /// -----------------------------------------------------------------
        JButton dLoad = ActionBuilder.create(COMMAND_LOAD, null, COMMAND_LOAD_MODEL_DESCRIPTION, KeyEvent.VK_L,
            new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    JFileChooser jFileChooser = new JFileChooser();
                    jFileChooser.setCurrentDirectory(new File(DEFAULT_LOAD_DIRECTORY));
                    jFileChooser.setFileFilter(new FileNameExtensionFilter("Geometric Constraint Model File", FILE_EXTENSION_GCM));
                    int response = jFileChooser.showOpenDialog(null);
                    if (response == JFileChooser.APPROVE_OPTION) {
                        FrameView.this.controller.readModelFrom(jFileChooser.getSelectedFile());
                    }
                }
            }, controllerEventQueue);
        /// -----------------------------------------------------------------
        JButton dStore = new JButton(COMMAND_STORE);
        JButton dClear = new JButton(COMMAND_CLEAR);
        JButton dNorm = new JButton(COMMAND_NORMAL1);
        JButton dLine = new JButton(COMMAND_DRAW_LINE);
        JButton dCircle = new JButton(COMMAND_DRAW_CIRCLE);
        JButton dArc = new JButton(COMMAND_DRAW_ARC);
        JButton dPoint = new JButton(COMMAND_DRAW_POINT);
        JButton dRefresh = new JButton(COMMAND_REFRESH);
        /// -----------------------------------------------------------------
        final JButton dSolve = ActionBuilder.create(COMMAND_SOLVE, null, COMMAND_SOLVE_DESCRIPTION, KeyEvent.VK_S,
            new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    runSolver(FrameView.this.controller);
                }
            }, controllerEventQueue);
        /// -----------------------------------------------------------------
        final JButton dReposition = ActionBuilder.create(COMMAND_REPOZ, null, COMMAND_REPOZ_DESCRIPTION, KeyEvent.VK_Z,
            new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    FrameView.this.controller.evaluateGuidePoints();
                    ms.refreshContainers();
                    ms.repaintLater();
                }
            }, controllerEventQueue);
        /// -----------------------------------------------------------------
        final JButton dRelaxe = ActionBuilder.create(COMMAND_RELAX, null, COMMAND_RELAX_DESCRIPTION, KeyEvent.VK_X,
            new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    FrameView.this.controller.relaxControlPoints();
                    FrameView.this.controller.evaluateGuidePoints();
                    ms.refreshContainers();
                    ms.repaintLater();
                }
            }, controllerEventQueue);
        /// -----------------------------------------------------------------

        JButton dCtrl = new JButton(COMMAND_CTRL);

        dSolve.setBackground(Color.GREEN);
        dCtrl.setBackground(Color.CYAN);
        /// -----------------------------------------------------------------

        dStore.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                JFileChooser jFileChooser = new JFileChooser();
                jFileChooser.setCurrentDirectory(new File(DEFAULT_LOAD_DIRECTORY));
                jFileChooser.setFileFilter(new FileNameExtensionFilter("Geometric Constraint Model File", FILE_EXTENSION_GCM));
                int response = jFileChooser.showSaveDialog(null);
                if (response == JFileChooser.APPROVE_OPTION) {
                    FrameView.this.controller.writeModelInto(jFileChooser.getSelectedFile());
                }
            }
        });

        dClear.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                ModelRegistry.removeObjectsFromModel();
                ms.refreshContainers();
                ms.repaintLater();
                clearTextArea();
                Events.send(EventType.REBUILD_TABLES, null);
            }
        });
        /// -----------------------------------------------------------------

        dNorm.addActionListener(e -> ms.setStateSketch(MySketchState.Normal));
        dLine.addActionListener(e -> ms.setStateSketch(MySketchState.DrawLine));
        dCircle.addActionListener(e -> ms.setStateSketch(MySketchState.DrawCircle));
        dArc.addActionListener(e -> ms.setStateSketch(MySketchState.DrawArc));
        dPoint.addActionListener(e -> ms.setStateSketch(MySketchState.DrawPoint));

        /// -----------------------------------------------------------------
        dRefresh.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                ms.refreshContainers();
                ms.repaintLater();
            }
        });
        /// -----------------------------------------------------------------
        dCtrl.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                ms.setControlGuidelines(!ms.isControlGuidelines());
                ms.repaintLater();
            }
        });

        /// -----------------------------------------------------------------
        Events.addListener(EventType.REFRESH_N_REPAINT, (eventType, arguments) -> {
            ms.refreshContainers();
            ms.repaintLater();
        });

        Events.addAwtListener(EventType.CONTROLLER_ERROR, (eventType, arguments) -> {
            JOptionPane.showMessageDialog(this, (String) arguments[0], "Application Error", JOptionPane.ERROR_MESSAGE);
        });

        /// -----------------------------------------------------------------
        ButtonGroup solverType = new ButtonGroup();
        JRadioButton onCPU = new JRadioButton("CPU colt", true);
        JRadioButton onGPU = new JRadioButton("GPU cuSolver", true);
        onCPU.setActionCommand(COMMAND_CPU);
        onGPU.setActionCommand(COMMAND_GPU);
        onCPU.setBackground(ON_CPU_COLOR);
        onGPU.setBackground(ON_GPU_COLOR);
        onCPU.setFocusable(false);
        onGPU.setFocusable(false);
        solverType.add(onCPU);
        solverType.add(onGPU);
        ActionListener solverActionListener = new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                final String actionCommand = e.getActionCommand();
                SwingUtilities.invokeLater(new Runnable() {
                    @Override
                    public void run() {
                        switch (actionCommand) {
                            case COMMAND_CPU:
                                FrameView.this.controller.setSolverType(GeometricSolverType.CPU_SOLVER);
                                break;
                            case COMMAND_GPU:
                                FrameView.this.controller.setSolverType(GeometricSolverType.GPU_SOLVER);
                                break;
                            default:
                                throw new Error("illegal solveAction action" + e.getActionCommand());
                        }
                    }
                });
            }
        };
        onCPU.addActionListener(solverActionListener);
        onGPU.addActionListener(solverActionListener);
        /// -----------------------------------------------------------------

        // FIXME - wazne dla setFocusable
        dLoad.setFocusable(false);
        dStore.setFocusable(false);
        dClear.setFocusable(false);
        dNorm.setFocusable(false);
        dLine.setFocusable(false);
        dCircle.setFocusable(false);
        dArc.setFocusable(false);
        dPoint.setFocusable(false);
        dRefresh.setFocusable(false);
        dSolve.setFocusable(false);
        dReposition.setFocusable(false);
        dRelaxe.setFocusable(false);
        dCtrl.setFocusable(false);

        jToolBar.add(dLoad);
        jToolBar.add(dStore);
        jToolBar.add(dClear);
        jToolBar.addSeparator(new Dimension(20, 1));
        jToolBar.add(dNorm);
        jToolBar.add(dLine);
        jToolBar.add(dCircle);
        jToolBar.add(dArc);
        jToolBar.add(dPoint);
        jToolBar.addSeparator(new Dimension(20, 1));
        jToolBar.add(dRefresh);
        jToolBar.addSeparator(new Dimension(20, 1));
        jToolBar.add(dSolve);
        jToolBar.addSeparator(new Dimension(20, 1));
        jToolBar.add(dReposition);
        jToolBar.add(dRelaxe);
        jToolBar.add(dCtrl);
        jToolBar.addSeparator(new Dimension(120, 1));
        jToolBar.add(onCPU);
        jToolBar.addSeparator(new Dimension(20, 1));
        jToolBar.add(onGPU);
        return jToolBar;
    }

    private JToolBar setupConstraintToolBar(Controller controller) {
        JToolBar toolbar = new JToolBar();
        toolbar.setLayout(new BoxLayout(toolbar, BoxLayout.Y_AXIS));
        JButton consHorizontal = new JButton("H");
        JButton consVertical = new JButton("V");
        JButton consPerpendicular = new JButton("L");
        JButton consPerallel = new JButton("LL");
        JButton consConnect2Points = new JButton("C2");
        JButton consTangency = new JButton("TG");
        JButton consEqualLength = new JButton("Eq");
        JButton consAngle2Lines = new JButton("A");
        JButton consCircleTangency = new JButton("OT");
        JButton consHorizontalPoint = new JButton("hP");
        JButton consVerticalPoint = new JButton("vP");
        JButton consFixPoint = new JButton("xP");

        consHorizontal.setMinimumSize(new Dimension(30, 30));
        consVertical.setMinimumSize(new Dimension(30, 30));

        toolbar.add(consHorizontal, Component.CENTER_ALIGNMENT );
        toolbar.add(consVertical , Component.CENTER_ALIGNMENT);
        toolbar.add(consPerpendicular);
        toolbar.add(consPerallel);
        toolbar.add(consConnect2Points);
        toolbar.add(consTangency);
        toolbar.add(consEqualLength);
        toolbar.add(consAngle2Lines);
        toolbar.add(consCircleTangency);
        toolbar.add(consHorizontalPoint);
        toolbar.add(consVerticalPoint);
        toolbar.add(consFixPoint);
        toolbar.setOrientation(VERTICAL);
        return toolbar;
    }


    private void runSolver(Controller controller) {
        controller.solveSystem();
        ms.refreshContainers();
        ms.repaintLater();
    }

    private void clearTextArea() {
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                consoleOutput.setText("");
                consoleOutput.setCaretPosition(consoleOutput.getDocument().getLength()); /// auto scroll - follow caret position
            }
        });
    }

    private final BlockingQueue<String> messageQueue = new LinkedBlockingQueue<>(64);

    private void updateTextArea(final String text) {

        try {
            messageQueue.put(text);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }

        if (messageQueue.size() == 0) {
            return;
        }

        SwingUtilities.invokeLater(new Runnable() {
            public void run() {

                if (messageQueue.isEmpty())
                    return;

                String text = null;
                while (true) {
                    try {
                        text = messageQueue.poll(90, TimeUnit.MILLISECONDS);
                    } catch (InterruptedException e) {
                        throw new RuntimeException(e);
                    }
                    if (text == null) {
                        break;
                    }
                    consoleOutput.append(text);
                }
                consoleOutput.setCaretPosition(consoleOutput.getDocument().getLength()); /// auto scroll - follow caret position
            }
        });
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
