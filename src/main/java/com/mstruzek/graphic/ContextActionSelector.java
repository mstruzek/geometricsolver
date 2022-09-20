package com.mstruzek.graphic;

import javax.swing.*;
import javax.swing.event.MouseInputAdapter;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;

public class ContextActionSelector extends MouseInputAdapter {

    private JPopupMenu popupMenu = new JPopupMenu();

    public ContextActionSelector() {
    }

    public void registerAction(String buttonLabel, ActionListener listener) {
        JMenuItem action = new JMenuItem(buttonLabel);
        action.addActionListener(listener);
        this.popupMenu.add(action);
    }

    @Override
    public void mouseReleased(MouseEvent e) {
        maybeShowPopup(e);
    }

    @Override
    public void mousePressed(MouseEvent e) {
        maybeShowPopup(e);
    }

    private void maybeShowPopup(MouseEvent e) {
        if (e.isPopupTrigger()) {
            popupMenu.show(e.getComponent(), e.getX(), e.getY());
        }

    }
}
