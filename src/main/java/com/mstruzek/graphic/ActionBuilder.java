package com.mstruzek.graphic;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.concurrent.ExecutorService;

public class ActionBuilder {

    public static JButton build(String name, Icon icon, String description, int keyEvent, ActionListener actionListener,
                                ExecutorService executorService) {

        AbstractAction action = new AbstractAction(name, icon) {
            @Override
            public void actionPerformed(ActionEvent e) {
                Runnable task = new Runnable() {
                    @Override
                    public void run() {
                        actionListener.actionPerformed(e);
                    }
                };
                executorService.submit(task);
            }
        };
        action.putValue(Action.SHORT_DESCRIPTION, description);
        action.putValue(Action.MNEMONIC_KEY, keyEvent);
        return new JButton(action);
    }
}
