package com.mstruzek.graphic;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;


import javax.swing.JLabel;
import javax.swing.JMenuItem;
import javax.swing.JPopupMenu;

public class MyPopup extends JPopupMenu implements ActionListener {


    private static final long serialVersionUID = 1L;
    /**
     * id aktualnie wskazanego punktu
     */
    int pointId;
    /**
     * kontener na przechwoywane punkty
     */
    MyPointContainer mpc = null;

    /**
     * Etykieta na ktorej piszemy dane
     */
    JLabel jl = null;

    public MyPopup(int pointId, MyPointContainer mpc, JLabel jl) {
        super();
        this.pointId = pointId;
        this.mpc = mpc;
        this.jl = jl;
    }

    public void rebuild() {
        this.removeAll();
        JLabel jl = new JLabel("Current Point"); // : p "+ this.pointId
        //JLabel jl = new JLabel("Current Point set to:" );
        JMenuItem miK = new JMenuItem("K");
            miK.addActionListener(this);
        JMenuItem miL = new JMenuItem("L");
            miL.addActionListener(this);
        JMenuItem miM = new JMenuItem("M");
            miM.addActionListener(this);
        JMenuItem miN = new JMenuItem("N");
            miN.addActionListener(this);
        JMenuItem ref = new JMenuItem("REFRESH");
            ref.addActionListener(this);
        JMenuItem clr = new JMenuItem("CLEAR");
            clr.addActionListener(this);
        this.add(jl);
        this.add(miK);
        this.add(miL);
        this.add(miM);
        this.add(miN);
        this.add(ref);
        this.add(clr);
    }

    public int getPointId() {
        return pointId;
    }


    public void setPointId(int pointId) {
        this.pointId = pointId;
    }


    @Override
    public void actionPerformed(ActionEvent e) {

        String what = ((JMenuItem) e.getSource()).getText();
        if (what.equals("K")) {
            mpc.setPointK(this.pointId);
        } else if (what.equals("L")) {
            mpc.setPointL(this.pointId);
        } else if (what.equals("M")) {
            mpc.setPointM(this.pointId);
        } else if (what.equals("N")) {
            mpc.setPointN(this.pointId);
        } else if (what.equals("REFRESH")) {
            this.jl.repaint();
        } else if (what.equals("CLEAR")) {
            mpc.clearAll();
        }

        this.jl.setText(mpc.toString());
    }


}
