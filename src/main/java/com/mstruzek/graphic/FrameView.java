package com.mstruzek.graphic;

import com.mstruzek.controller.ActiveKey;
import com.mstruzek.controller.Controller;
import com.mstruzek.controller.EventType;
import com.mstruzek.controller.Events;
import com.mstruzek.msketch.GeometricConstraintType;

import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Objects;
import java.util.Random;
import java.util.concurrent.CompletableFuture;

import static com.mstruzek.graphic.Property.CURRENT_POSITION;
import static com.mstruzek.graphic.Property.KLMN_POINTS;

public class FrameView extends JFrame {

    private static final long serialVersionUID = 1L;
    public static final int L_WIDTH = 920;
    public static final int R_HEIGHT = 1400;
    public static final int L_HEIGHT = 1400;
    public static final int R_WIDTH = 1420;

    private Container pane = getContentPane();

    /**
     * zmienna na parametry
     */
    final JTextField param = new JTextField(5);

    {
        param.setText("10.0");
    }

    /**
     * tabelki z wiezami,figurami i parametrami
     */
    final MyTables myTables;

    /**
     * wypisujemy tutaj wszystko co idzie na System.out.println();
     */
    final JTextArea sytemOutPrintln = new JTextArea(25, 85);
    /**
     * glowny controller
     */
    final Controller controller;

    // widok na pojemnik K,L,M,N
    final JLabel klmn = new JLabel("K,L,M,N", SwingConstants.CENTER);
    ;

    // wyswietla aktualna pozycje kursora
    final JLabel currentPosition = new JLabel("Currrent Position:");

    final MySketch ms;

    public FrameView(String windowTitle, Controller controller) {
        super(windowTitle);
        this.controller = controller;

        pane.setLayout(new BorderLayout());

        setFocusable(true);
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

        // OGOLNY ROZKLAD
        JPanel main = new JPanel();
        main.setLayout(new BoxLayout(main, BoxLayout.X_AXIS));

        JPanel left = new JPanel();
        JPanel right = new JPanel();

        left.setPreferredSize(new Dimension(L_WIDTH, L_HEIGHT));
        right.setPreferredSize(new Dimension(R_WIDTH, R_HEIGHT));

        left.setBorder(BorderFactory.createLineBorder(Color.CYAN));
        right.setBorder(BorderFactory.createLineBorder(Color.CYAN));

        main.add(left);
        main.add(right);

        // SZKICOWNIK
        ms = new MySketch(this.controller);
        ms.setWithControlLines(false);
        ms.addPropertyChangeListener(new PropertyChangeListener() {
            @Override
            public void propertyChange(PropertyChangeEvent evt) {
                switch (evt.getPropertyName()) {
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
        panPoints.setBackground(new Color(250, 200, 200));
        panPoints.setPreferredSize(new Dimension(920, 50));
        panPoints.setBorder(BorderFactory.createLineBorder(Color.black));
        panPoints.add(klmn, BorderLayout.NORTH);
        panPoints.add(currentPosition, BorderLayout.SOUTH);

        left.add(panPoints);

        // Dodawanie wiezow

        JPanel constMenu = new JPanel();
        constMenu.setBackground(new Color(244, 249, 192));
        constMenu.setPreferredSize(new Dimension(400, 240));
        constMenu.setBorder(BorderFactory.createTitledBorder("Add Constraint"));

        final JTextArea consDescr = new JTextArea(7, 30);
        consDescr.setBorder(BorderFactory.createTitledBorder("HELP"));
        consDescr.setLineWrap(true);
        consDescr.setWrapStyleWord(true);
        consDescr.setEditable(false);
        consDescr.setFocusable(false);
        consDescr.setBackground(new Color(100, 255, 100, 50));

        final JComboBox combo = new JComboBox(GeometricConstraintType.values());
        combo.setFocusable(false);
        combo.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                JComboBox cb = (JComboBox) e.getSource();
                GeometricConstraintType what = (GeometricConstraintType) cb.getSelectedItem();
                consDescr.setText(Objects.requireNonNull(what).getHelp());
                if(consDescr.getParent() != null) {
                    consDescr.getParent().repaint();
                }
            }
        });

        combo.setSelectedItem(GeometricConstraintType.FixPoint);

        JButton addCons = new JButton("Add Constraint");
        addCons.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent e) {
                GeometricConstraintType constraintType = (GeometricConstraintType) combo.getSelectedItem();

                if(constraintType == null) {
                    throw new Error("constraint type invalid !");
                }
                FrameView.this.controller.addConstraint(constraintType,
                    ms.getPointK(),
                    ms.getPointL(),
                    ms.getPointM(),
                    ms.getPointN(),
                    Double.parseDouble(param.getText())
                );
                ms.clearAll();
                requestFocus();
            }
        });
        constMenu.add(combo);
        constMenu.add(consDescr);
        constMenu.add(param);
        constMenu.add(addCons);

        left.add(constMenu);

        // Tabelki
        myTables = new MyTables();
        myTables.setFocusable(false);

        right.add(myTables);
        right.add(new JScrollPane(sytemOutPrintln));
        redirectSystemStreams();

        // ToolBar
        JToolBar jToolBar = new JToolBar();
        JButton dload = new JButton("Load");
        JButton dstore = new JButton("Store");
        JButton dnorm = new JButton("Normal");
        JButton dline = new JButton("Draw Line");
        JButton dcircle = new JButton("Draw Circle");
        JButton darc = new JButton("Draw Arc");
        JButton dpoint = new JButton("Draw Point");
        JButton drefresh = new JButton("REFRESH");
        JButton dsolve = new JButton("SOLVE");
        JButton drelax = new JButton("RELAX");
        JButton dfluctuate = new JButton("FLUCTUATE");
        JButton dcontrol = new JButton("CTRL");
        dsolve.setBackground(Color.GREEN);
        dcontrol.setBackground(Color.CYAN);

        dload.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                JFileChooser jFileChooser = new JFileChooser();
                jFileChooser.setFileFilter(new FileNameExtensionFilter("Geometric Constraint Model File", "gcm"));
                int response = jFileChooser.showSaveDialog(null);
                if(response == JFileChooser.APPROVE_OPTION) {
                    controller.readModelFrom(jFileChooser.getSelectedFile());
                }
            }
        });

        dstore.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                JFileChooser jFileChooser = new JFileChooser();
                jFileChooser.setFileFilter(new FileNameExtensionFilter("Geometric Constraint Model File", "gcm"));
                int response = jFileChooser.showSaveDialog(null);
                if(response == JFileChooser.APPROVE_OPTION) {
                    controller.writeModelInto(jFileChooser.getSelectedFile());
                }
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
        });

        drelax.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                controller.relaxForces();
                ms.refreshContainers();
                ms.repaint();
            }
        });

        dfluctuate.addActionListener(new ActionListener() {

            private final Random random = new Random();

            @Override
            public void actionPerformed(ActionEvent e) {
                double coefficient = random.nextDouble() * 20;
                controller.fluctuatePoints(coefficient);
                ms.refreshContainers();
                ms.repaint();
            }
        });

        dcontrol.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                ms.setWithControlLines(!ms.isWithControlLines());
                ms.repaint();
            }
        });

        Events.addListener(EventType.REFRESH_N_REPAINT, (eventType, arguments) -> {
            ms.refreshContainers();
            ms.repaint();
        });

        // FIXME - wazne dla setFocusable
        dload.setFocusable(false);
        dstore.setFocusable(false);
        dnorm.setFocusable(false);
        dline.setFocusable(false);
        dcircle.setFocusable(false);
        darc.setFocusable(false);
        dpoint.setFocusable(false);
        drefresh.setFocusable(false);
        dsolve.setFocusable(false);
        drelax.setFocusable(false);
        dfluctuate.setFocusable(false);
        dcontrol.setFocusable(false);

        jToolBar.add(dload);
        jToolBar.add(dstore);
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
        jToolBar.add(drelax);
        jToolBar.addSeparator(new Dimension(20, 1));
        jToolBar.add(dfluctuate);
        jToolBar.add(dcontrol);

        // GLOWNY ROZKLAD TOOLBAR I OKNO
        pane.add(jToolBar, BorderLayout.NORTH);
        pane.add(main, BorderLayout.CENTER);

        // KONIEC
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        pack();
        setVisible(true);
    }

    private void updateTextArea(final String text) {
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                sytemOutPrintln.append(text);
            }
        });
    }

    private void redirectSystemStreams() {
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
