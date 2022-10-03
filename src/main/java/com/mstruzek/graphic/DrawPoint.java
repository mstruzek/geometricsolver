package com.mstruzek.graphic;

import com.mstruzek.msketch.ModelRegistry;

import java.awt.*;

public class DrawPoint extends Point {

    int id;
    boolean hover = false;

    public DrawPoint(int p) {
        id = p;
    }

    public boolean hover() {
        return hover;
    }

    public void setHover(boolean hover) {
        this.hover = hover;
    }

    @Override
    public void setLocation(double x, double y) {
        com.mstruzek.msketch.Point p = ModelRegistry.dbPoint.get(this.id);
        p.setX(x);
        p.setY(y);
    }

    @Override
    public void setLocation(Point ps) {
        super.setLocation(ps);
        com.mstruzek.msketch.Point p = ModelRegistry.dbPoint.get(this.id);
        p.setX(ps.getX());
        p.setY(ps.getY());
    }

    @Override
    public java.awt.Point getLocation() {
        com.mstruzek.msketch.Point point = ModelRegistry.dbPoint.get(this.id);
        return new java.awt.Point((int) point.getX(), (int) point.getY());
    }
}
