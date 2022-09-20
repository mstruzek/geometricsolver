package com.mstruzek;

import com.mstruzek.controller.Controller;
import com.mstruzek.graphic.FrameView;

import javax.swing.*;
import java.awt.*;

public class AppBootstrap {

    public static void main(String[] args) {
        /*
         * Glowny kontroller widoku swingowego.
         */
        Controller controller = new Controller();

        FrameView frameView = new FrameView(controller);
        // full screen width
        frameView.setMaximizedBounds(new Rectangle(Integer.MAX_VALUE, Integer.MAX_VALUE));
        frameView.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frameView.pack();
        /// operation order `full screen width
        frameView.setExtendedState(frameView.getExtendedState() | JFrame.MAXIMIZED_BOTH);
        frameView.setVisible(true);

        Runtime.getRuntime().addShutdownHook(new Thread(new Runnable() {
            @Override
            public void run() {
                controller.shutdown();
            }
        }));
    }
}
