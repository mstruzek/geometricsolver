package com.mstruzek.graphic;

public class DrawArc extends DrawElement {

    DrawPoint p1, p2, p3;

    public DrawArc(DrawPoint tp1, DrawPoint tp2, DrawPoint tp3) {
        p1 = tp1;
        p2 = tp2;
        p3 = tp3;
    }
}