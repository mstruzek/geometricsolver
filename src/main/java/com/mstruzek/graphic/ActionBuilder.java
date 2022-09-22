package com.mstruzek.graphic;

import com.mstruzek.utils.Dispatcher;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class ActionBuilder {

    public static JButton create(String name, Icon icon, String description, int keyEvent, ActionListener actionListener,
                                 Dispatcher dispatcher) {

        AbstractAction action = new AbstractAction(name, icon) {
            @Override
            public void actionPerformed(ActionEvent e) {
                dispatcher.submit(() -> actionListener.actionPerformed(e));
            }
        };
        action.putValue(Action.SHORT_DESCRIPTION, description);
        action.putValue(Action.MNEMONIC_KEY, keyEvent);
        return new JButton(action);
    }
}
