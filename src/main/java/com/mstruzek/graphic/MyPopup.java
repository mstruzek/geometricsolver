package com.mstruzek.graphic;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;


import javax.swing.JLabel;
import javax.swing.JMenuItem;
import javax.swing.JPopupMenu;

import static com.mstruzek.graphic.Property.KLMN_POINTS;

public class MyPopup extends JPopupMenu implements ActionListener {

    /**
     * id aktualnie wskazanego punktu
     */
    int pointId;
    /**
     * kontener na przechwoywane punkty
     */
    final MyPointContainer mpc;

    public MyPopup(int pointId, MyPointContainer mpc) {
        super();
        this.pointId = pointId;
        this.mpc = mpc;
        build();
    }

    public void build() {
        JLabel jl = new JLabel("Current Point"); // : p "+ this.pointId
        JMenuItem miK = new JMenuItem("K");
        miK.addActionListener(this);
        JMenuItem miL = new JMenuItem("L");
        miL.addActionListener(this);
        JMenuItem miM = new JMenuItem("M");
        miM.addActionListener(this);
        JMenuItem miN = new JMenuItem("N");
        miN.addActionListener(this);
        JMenuItem clr = new JMenuItem("CLEAR");
        clr.addActionListener(this);
        this.add(jl);
        this.add(miK);
        this.add(miL);
        this.add(miM);
        this.add(miN);
        this.add(clr);
    }

    public void setPointId(int pointId) {
        this.pointId = pointId;
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        String what = ((JMenuItem) e.getSource()).getText();
        switch(what){
            case "K":
                mpc.setPointK(this.pointId);
                break;
            case "L":
                mpc.setPointL(this.pointId);
                break;
            case "M":
                mpc.setPointM(this.pointId);
                break;
            case "N":
                mpc.setPointN(this.pointId);
                break;
            case "CLEAR":
                mpc.clearAll();
                break;
        }
        firePropertyChange(KLMN_POINTS, null, mpc.toString());
    }

}
