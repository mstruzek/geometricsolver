package com.mstruzek.graphic;

import com.mstruzek.controller.EventType;
import com.mstruzek.controller.Events;
import com.mstruzek.msketch.solver.SolverStat;

import javax.swing.*;
import java.awt.*;

public class SolverStatPanel extends JPanel {
    private JTextField f_startTime              = new JTextField(); 
    private JTextField f_stopTime               = new JTextField(); 
    private JTextField f_size                   = new JTextField(); 
    private JTextField f_coefficientArity       = new JTextField(); 
    private JTextField f_dimension              = new JTextField(); 
    private JTextField f_accEvaluationTime      = new JTextField(); 
    private JTextField f_accSolverTime          = new JTextField(); 
    private JTextField f_convergence            = new JTextField(); 
    private JTextField f_delta                  = new JTextField(); 
    private JTextField f_constraintDelta        = new JTextField(); 
    private JTextField f_iterations             = new JTextField(); 
    private JTextField f_accTime                = new JTextField();

    private JLabel l_startTime              = new JLabel("Start Time [ns]: ");
    private JLabel l_stopTime               = new JLabel("Stop Time [ns]: ");
    private JLabel l_size                   = new JLabel("Size: ");
    private JLabel l_coefficientArity       = new JLabel("Coefficient Size: ");
    private JLabel l_dimension              = new JLabel("Dimension: ");
    private JLabel l_accEvaluationTime      = new JLabel("Time AccEvaluation [ns]: ");
    private JLabel l_accSolverTime          = new JLabel("Time AccSolver [ns]: ");
    private JLabel l_convergence            = new JLabel("Convergence: ");
    private JLabel l_delta                  = new JLabel("Error: ");
    private JLabel l_constraintDelta        = new JLabel("ConstraintDelta: ");
    private JLabel l_iterations             = new JLabel("iter-n: ");
    private JLabel l_accTime                = new JLabel("Acc Time [ns]: ");

    private static final String FORMAT_DECIMAL  = "  %d ";
    private static final String FORMAT_STR      = "  %s ";
    private static final String FORMAT_TIME     = "  %,12d  ";
    private static final String FORMAT_DELTA    = "  %e ";

    public SolverStatPanel() {
        super();
        GroupLayout layout = new GroupLayout(this);
        this.setLayout(layout);
        // this.setBackground(Color.lightGray);
        this.setBorder(BorderFactory.createTitledBorder(BorderFactory.createLineBorder(Color.DARK_GRAY), "Stats"));

        lightedStyle(f_accEvaluationTime);
        lightedStyle(f_accSolverTime);
        lightedStyle(f_convergence);
        lightedStyle(f_delta);

        layout.setAutoCreateGaps(true);
        layout.setHorizontalGroup(
            layout.createSequentialGroup()
                .addGroup(layout.createParallelGroup(GroupLayout.Alignment.LEADING)
                    .addComponent(l_startTime)
                    .addComponent(l_stopTime)
                    .addComponent(l_size)
                    .addComponent(l_coefficientArity))
                .addGroup(layout.createParallelGroup(GroupLayout.Alignment.LEADING)
                    .addComponent(f_startTime)
                    .addComponent(f_stopTime)
                    .addComponent(f_size)
                    .addComponent(f_coefficientArity))
                .addGroup(layout.createParallelGroup(GroupLayout.Alignment.LEADING)
                    .addComponent(l_dimension)
                    .addComponent(l_accEvaluationTime)
                    .addComponent(l_accSolverTime)
                    .addComponent(l_convergence))
                .addGroup(layout.createParallelGroup(GroupLayout.Alignment.LEADING)
                    .addComponent(f_dimension)
                    .addComponent(f_accEvaluationTime)
                    .addComponent(f_accSolverTime)
                    .addComponent(f_convergence))
                .addGroup(layout.createParallelGroup(GroupLayout.Alignment.LEADING)
                    .addComponent(l_delta)
                    .addComponent(l_constraintDelta)
                    .addComponent(l_iterations)
                    .addComponent(l_accTime))
                .addGroup(layout.createParallelGroup(GroupLayout.Alignment.LEADING)
                    .addComponent(f_delta)
                    .addComponent(f_constraintDelta)
                    .addComponent(f_iterations)
                    .addComponent(f_accTime))
        );

        layout.setVerticalGroup(
            layout.createSequentialGroup()
                .addGroup(layout.createParallelGroup()
                    .addComponent(l_startTime)
                    .addComponent(f_startTime)
                    .addComponent(l_dimension)
                    .addComponent(f_dimension)
                    .addComponent(l_delta)
                    .addComponent(f_delta))
                .addGroup(layout.createParallelGroup()
                    .addComponent(l_stopTime)
                    .addComponent(f_stopTime)
                    .addComponent(l_accEvaluationTime)
                    .addComponent(f_accEvaluationTime)
                    .addComponent(l_constraintDelta)
                    .addComponent(f_constraintDelta))
                .addGroup(layout.createParallelGroup()
                    .addComponent(l_size)
                    .addComponent(f_size)
                    .addComponent(l_accSolverTime)
                    .addComponent(f_accSolverTime)
                    .addComponent(l_iterations)
                    .addComponent(f_iterations))
                .addGroup(layout.createParallelGroup()
                    .addComponent(l_coefficientArity)
                    .addComponent(f_coefficientArity)
                    .addComponent(l_convergence)
                    .addComponent(f_convergence)
                    .addComponent(l_accTime)
                    .addComponent(f_accTime))
        );

        Events.addListener(EventType.SOLVER_STAT_CHANGE, new Events.EventHandler() {
            @Override
            public void call(String eventType, Object[] arguments) {
                SolverStat stat = (SolverStat) arguments[0];
                f_startTime.setText(String.format(FORMAT_TIME, stat.startTime));
                f_stopTime.setText(String.format(FORMAT_TIME, stat.stopTime));
                f_size.setText(String.format(FORMAT_DECIMAL, stat.size));
                f_coefficientArity.setText(String.format(FORMAT_DECIMAL, stat.coefficientArity));
                f_dimension.setText(String.format(FORMAT_DECIMAL, stat.dimension));
                f_accEvaluationTime.setText(String.format(FORMAT_TIME, stat.accEvaluationTime));
                f_accSolverTime.setText(String.format(FORMAT_TIME, stat.accSolverTime));
                f_convergence.setText(String.format(FORMAT_STR, stat.convergence));
                f_delta.setText(String.format(FORMAT_DELTA, stat.error));
                f_constraintDelta.setText(String.format(FORMAT_DELTA, stat.constraintDelta));
                f_iterations.setText(String.format(FORMAT_DECIMAL, stat.iterations));
                f_accTime.setText(String.format(FORMAT_TIME, (stat.stopTime - stat.startTime)));
            }
        });
    }

    /***
     * Highlighted most important data points
     * @param textField style applied
     */
    private static void lightedStyle(JTextField textField) {
        textField.setBackground(Color.LIGHT_GRAY);
//        textField.setBorder(BorderFactory.createLineBorder(Color.RED));
    }

}

///1

