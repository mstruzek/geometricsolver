package com.mstruzek;

import com.mstruzek.controller.Controller;
import com.mstruzek.msketch.ConstraintType;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;

import static javax.swing.SwingConstants.VERTICAL;

public class Main {


    public static class TestView extends JFrame {

        public static final int WIDTH1 = 60;

        public TestView(String title) throws HeadlessException {
            super(title);


            JPanel jPanel = new JPanel();

            jPanel.setBackground(Color.green);
            jPanel.setMinimumSize(new Dimension(WIDTH1, 400));
            jPanel.setMaximumSize(new Dimension(WIDTH1, 400));


            setContentPane(setupConstraintToolBar(null));

        }

        private JToolBar setupConstraintToolBar(Controller controller) {
            JToolBar toolbar = new JToolBar();
            toolbar.setBorder(BorderFactory.createLineBorder(Color.green, 1));
            BoxLayout mgr = new BoxLayout(toolbar, BoxLayout.Y_AXIS);
            toolbar.setLayout(mgr);
            toolbar.setAlignmentX(Component.CENTER_ALIGNMENT);

            ActionListener actionListener = new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    ConstraintType constraintType = ConstraintType.valueOf(e.getActionCommand());

                    System.out.printf("+" + constraintType.name() + "\n");
                }
            };
            class ComponentBuilder {
                public JButton newButton(String title, int mnemonic, String actionCommand) {
                    JButton button = new JButton(title);
                    button.setMnemonic(mnemonic);
                    button.setActionCommand(actionCommand);
                    button.addActionListener(actionListener);
                    button.setAlignmentX(Component.CENTER_ALIGNMENT);
                    button.setMinimumSize(new Dimension(WIDTH1,30));
                    button.setSize(new Dimension(WIDTH1,30));
                    button.setMaximumSize(new Dimension(WIDTH1,30));
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
    }

    public static void main(String[] args) {


        TestView frame = new TestView("test views");
        frame.pack();
        frame.setVisible(true);

    }
}