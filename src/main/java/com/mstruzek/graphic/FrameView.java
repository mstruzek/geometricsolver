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
import java.util.Objects;
import java.util.Random;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

import static com.mstruzek.graphic.Property.*;

public class FrameView extends JFrame {

    private static final long serialVersionUID = 1L;
    public static final int L_WIDTH = 920;
    public static final int R_HEIGHT = 1400;
    public static final int L_HEIGHT = 1400;
    public static final int R_WIDTH = 1420;

    public static final int CONSOLE_WIDTH = 920;
    public static final int CONSOLE_OUTPUT_HEIGHT = 420;


    public static final Color ON_CPU_COLOR = new Color(0,  181, 245);
    public static final Color ON_GPU_COLOR = new Color(118,185, 0);
    public static final Color CONSTRAINT_BORDER_COLOR = Color.DARK_GRAY;
    public static final Color CONSTRAINT_PANEL_BACKGROUND_COLOR = new Color(244, 249, 192);
    public static final Color HELP_PANEL_BACKGROUND_COLOR = new Color(100, 255, 100, 50);
    public static final Color SKETCH_INFO_BORDER_COLOR = Color.darkGray;
    public static final Color SKETCH_INFO_BACKGROUND_COLOR = new Color(250, 200, 200);
    public static final Color FRAME_BACKGROUND_COLOR = null; /// Default Wash
    public static final String FRAME_TITLE = "GCS GeometricSolver 2009-2022";
    public static final int SOLVER_PANEL_HEIGHT = 140;
    public static final int SOLVER_PANEL_WIDTH = 920;

    /*
     * Toolbar Actions
     */
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
    public static final String COMMAND_REPOS = "Repos";
    public static final String COMMAND_RELAX = "Relax";
    public static final String COMMAND_CTRL = "CTRL";
    public static final String COMMAND_CPU = "CPU";
    public static final String COMMAND_GPU = "GPU";


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

    // widok na pojemnik K,L,M,N
    final JLabel klmn = new JLabel("K,L,M,N", SwingConstants.CENTER);

    final JLabel pickedPoint = new JLabel("", SwingConstants.CENTER);
    ;

    // wyswietla aktualna pozycje kursora
    final JLabel currentPosition = new JLabel("Currrent Position:");

    final MySketch ms;

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

            @Override
            public void keyReleased(KeyEvent e) {

                if (e.isControlDown() && e.getKeyCode() == 88) { /// key combination :  [ CTRL + X ]
                    /// run solver round
                    runSolver(controller);
                }
                super.keyReleased(e);
            }
        });

        // OGOLNY ROZKLAD
        JPanel main = new JPanel();
        main.setLayout(new BoxLayout(main, BoxLayout.X_AXIS));

        JPanel left = new JPanel();
        JPanel right = new JPanel();

        left.setPreferredSize(new Dimension(L_WIDTH, L_HEIGHT));
        right.setPreferredSize(new Dimension(R_WIDTH, R_HEIGHT));

        left.setBackground(FRAME_BACKGROUND_COLOR);
        right.setBackground(FRAME_BACKGROUND_COLOR);

        ///  left.setBorder(BorderFactory.createLineBorder(Color.CYAN));
        /// right.setBorder(BorderFactory.createLineBorder(Color.CYAN));

        main.add(left);
        main.add(right);

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

        JPanel panPoints = new JPanel();
        panPoints.setLayout(new BorderLayout());
        panPoints.setBackground(SKETCH_INFO_BACKGROUND_COLOR);
        panPoints.setPreferredSize(new Dimension(SOLVER_PANEL_WIDTH, 60));
        panPoints.setBorder(BorderFactory.createLineBorder(SKETCH_INFO_BORDER_COLOR));
        panPoints.add(klmn, BorderLayout.NORTH);
        panPoints.add(currentPosition, BorderLayout.SOUTH);
        panPoints.add(pickedPoint, BorderLayout.EAST);

        left.add(panPoints);

        // Dodawanie wiezow

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

        JButton constraintButton = new JButton("Add Constraint");
        constraintButton.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent e) {
                ConstraintType constraintType = (ConstraintType) combo.getSelectedItem();

                if (constraintType == null) {
                    throw new Error("constraint type invalid !");
                }
                FrameView.this.controller.addConstraint(constraintType,
                    ms.getPointK(),
                    ms.getPointL(),
                    ms.getPointM(),
                    ms.getPointN(),
                    Double.parseDouble(parameterField.getText())
                );
                ms.clearAll();
                requestFocus();
            }
        });

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


        left.add(constraintPanel);

        // Tabelki
        myTables = new MyTables();
        myTables.setFocusable(false);

        right.add(myTables);

        consoleOutput.setFont(new Font("Courier New", Font.PLAIN, 12));
        consoleScrollPane = new JScrollPane(consoleOutput);
        consoleScrollPane.setPreferredSize(new Dimension(CONSOLE_WIDTH, CONSOLE_OUTPUT_HEIGHT));
        consoleScrollPane.scrollRectToVisible(consoleOutput.getVisibleRect());
        //redirectStdErrOut();

        SolverStatPanel solverStatPanel = new SolverStatPanel();
        solverStatPanel.setPreferredSize(new Dimension(SOLVER_PANEL_WIDTH, SOLVER_PANEL_HEIGHT));
        right.add(solverStatPanel);
        right.add(consoleScrollPane);

        // ToolBar
        JToolBar jToolBar = new JToolBar();
        JButton dload = new JButton(COMMAND_LOAD);
        JButton dstore = new JButton(COMMAND_STORE);
        JButton dclear = new JButton(COMMAND_CLEAR);
        JButton dnorm = new JButton(COMMAND_NORMAL1);
        JButton dline = new JButton(COMMAND_DRAW_LINE);
        JButton dcircle = new JButton(COMMAND_DRAW_CIRCLE);
        JButton darc = new JButton(COMMAND_DRAW_ARC);
        JButton dpoint = new JButton(COMMAND_DRAW_POINT);
        JButton drefresh = new JButton(COMMAND_REFRESH);
        JButton dsolve = new JButton(COMMAND_SOLVE);
        JButton dreposition = new JButton(COMMAND_REPOS);
        JButton drelaxe = new JButton(COMMAND_RELAX);
        JButton dctrl = new JButton(COMMAND_CTRL);

        JRadioButton onCPU = new JRadioButton("CPU colt", true);
        JRadioButton onGPU = new JRadioButton("GPU cuSolver", true);
        onCPU.setActionCommand(COMMAND_CPU);
        onGPU.setActionCommand(COMMAND_GPU);
        onCPU.setBackground(ON_CPU_COLOR);
        onGPU.setBackground(ON_GPU_COLOR);


        dsolve.setBackground(Color.GREEN);
        dctrl.setBackground(Color.CYAN);

        dload.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                JFileChooser jFileChooser = new JFileChooser();
                jFileChooser.setCurrentDirectory(new File("e:\\source\\gsketcher\\data\\"));
                jFileChooser.setFileFilter(new FileNameExtensionFilter("Geometric Constraint Model File", "gcm"));
                int response = jFileChooser.showOpenDialog(null);
                if (response == JFileChooser.APPROVE_OPTION) {
                    controller.readModelFrom(jFileChooser.getSelectedFile());
                }
            }
        });

        dstore.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                JFileChooser jFileChooser = new JFileChooser();
                jFileChooser.setCurrentDirectory(new File("e:\\source\\gsketcher\\data\\"));
                jFileChooser.setFileFilter(new FileNameExtensionFilter("Geometric Constraint Model File", "gcm"));
                int response = jFileChooser.showSaveDialog(null);
                if (response == JFileChooser.APPROVE_OPTION) {
                    controller.writeModelInto(jFileChooser.getSelectedFile());
                }
            }
        });
        dclear.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                ModelRegistry.removeObjectsFromModel();
                ms.refreshContainers();
                ms.repaint();
                clearTextArea();
                Events.send(EventType.REBUILD_TABLES, null);
            }
        });

        dnorm.addActionListener(e -> ms.setStateSketch(MySketchState.Normal));
        dline.addActionListener(e -> ms.setStateSketch(MySketchState.DrawLine));
        dcircle.addActionListener(e -> ms.setStateSketch(MySketchState.DrawCircle));
        darc.addActionListener(e -> ms.setStateSketch(MySketchState.DrawArc));
        dpoint.addActionListener(e -> ms.setStateSketch(MySketchState.DrawPoint));

        drefresh.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                ms.refreshContainers();
                ms.repaint();
            }
        });

        dsolve.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                runSolver(controller);
            }
        });

        dreposition.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                controller.evaluateGuidePoints();
                ms.refreshContainers();
                ms.repaint();
            }
        });

        drelaxe.addActionListener(new ActionListener() {

            private final Random random = new Random();

            @Override
            public void actionPerformed(ActionEvent e) {
                double scale = random.nextDouble() / 7;
                controller.relaxControlPoints(scale);
                ms.refreshContainers();
                ms.repaint();
            }
        });

        dctrl.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                ms.setControlGuidelines(!ms.isControlGuidelines());
                ms.repaint();
            }
        });

        Events.addListener(EventType.REFRESH_N_REPAINT, (eventType, arguments) -> {
            ms.refreshContainers();
            ms.repaint();
        });

        Events.addListener(EventType.CONTROLLER_ERROR, (eventType, arguments) -> {
            SwingUtilities.invokeLater(() -> {
                JOptionPane.showMessageDialog(this, (String) arguments[0], "Application Error", JOptionPane.ERROR_MESSAGE);

            });
        });

        ActionListener solverActionListener = new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                final String actionCommand = e.getActionCommand();
                SwingUtilities.invokeLater(new Runnable() {
                    @Override
                    public void run() {
                        final GeometricSolverType solverType = switch (actionCommand) {
                            case COMMAND_CPU -> GeometricSolverType.CPU_SOLVER;
                            case COMMAND_GPU -> GeometricSolverType.GPU_SOLVER;
                            default -> throw new Error("illegal solver action" + e.getActionCommand());
                        };
                        controller.setSolverType(solverType);
                    }
                });
            }
        };
        onCPU.addActionListener(solverActionListener);
        onGPU.addActionListener(solverActionListener);

        ButtonGroup solverType = new ButtonGroup();
        solverType.add(onCPU);
        solverType.add(onGPU);

        onCPU.setFocusable(false);
        onGPU.setFocusable(false);


        // FIXME - wazne dla setFocusable
        dload.setFocusable(false);
        dstore.setFocusable(false);
        dclear.setFocusable(false);
        dnorm.setFocusable(false);
        dline.setFocusable(false);
        dcircle.setFocusable(false);
        darc.setFocusable(false);
        dpoint.setFocusable(false);
        drefresh.setFocusable(false);
        dsolve.setFocusable(false);
        dreposition.setFocusable(false);
        drelaxe.setFocusable(false);
        dctrl.setFocusable(false);

        jToolBar.add(dload);
        jToolBar.add(dstore);
        jToolBar.add(dclear);
        jToolBar.addSeparator(new Dimension(20, 1));
        jToolBar.add(dnorm);
        jToolBar.add(dline);
        jToolBar.add(dcircle);
        jToolBar.add(darc);
        jToolBar.add(dpoint);
        jToolBar.addSeparator(new Dimension(20, 1));
        jToolBar.add(drefresh);
        jToolBar.addSeparator(new Dimension(20, 1));
        jToolBar.add(dsolve);
        jToolBar.addSeparator(new Dimension(20, 1));
        jToolBar.add(dreposition);
        jToolBar.add(drelaxe);
        jToolBar.add(dctrl);
        jToolBar.addSeparator(new Dimension(120, 1));
        jToolBar.add(onCPU);
        jToolBar.addSeparator(new Dimension(20, 1));
        jToolBar.add(onGPU);


        // GLOWNY ROZKLAD TOOLBAR I OKNO
        pane.add(jToolBar, BorderLayout.NORTH);
        pane.add(main, BorderLayout.CENTER);

        // KONIEC
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        pack();
        setVisible(true);
    }

    private void runSolver(Controller controller) {
        CompletableFuture.runAsync(controller::solveSystem)
            .thenRunAsync(() -> {
                ms.refreshContainers();
                SwingUtilities.invokeLater(new Runnable() {
                    @Override
                    public void run() {
                        ms.repaint();
                    }
                });
            });
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
