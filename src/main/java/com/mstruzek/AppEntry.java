package com.mstruzek;

import com.mstruzek.controller.Controller;
import com.mstruzek.graphic.TView;

public class AppEntry {

    public static void main(String[] args) {


        /*
        * Glowny kontroller widoku swingowego. <= hUNd
        */
        Controller controller = new Controller();

        new TView("M-Sketcher", controller);

    }
}
