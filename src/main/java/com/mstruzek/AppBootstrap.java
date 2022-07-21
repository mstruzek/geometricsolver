package com.mstruzek;

import com.mstruzek.controller.Controller;
import com.mstruzek.graphic.FrameView;

public class AppBootstrap {

    public static void main(String[] args) {
        /*
         * Glowny kontroller widoku swingowego.
         */
        Controller controller = new Controller();

        new FrameView(controller);
    }
}
