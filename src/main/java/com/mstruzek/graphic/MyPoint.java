package com.mstruzek.graphic;

import com.mstruzek.msketch.ModelRegistry;

import java.awt.*;
import java.awt.geom.Ellipse2D;


public class MyPoint extends Point {

    boolean hover = false;
    boolean dragged = false;
    int id;

    public MyPoint(int p) {
        id = p;
    }

    public boolean contains(int ix, int iy, double d) {
        com.mstruzek.msketch.Point p = ModelRegistry.dbPoint.get(this.id);
        Ellipse2D.Double circle = new Ellipse2D.Double(p.getX() - d / 2, p.getY() - d / 2, d, d);
        return circle.contains(ix, iy);
    }

    public boolean isDragged() {
        return dragged;
    }

    public void setDragged(boolean dragged) {
        this.dragged = dragged;
    }

    public boolean hover() {
        return hover;
    }

    public void setHover(boolean hover) {
        this.hover = hover;
    }

    @Override
    public double getX() {
        return ModelRegistry.dbPoint.get(this.id).getX();
    }

    @Override
    public double getY() {
        return ModelRegistry.dbPoint.get(this.id).getY();
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

    public java.awt.Point getLocation() {
        return new java.awt.Point((int) getX(), (int) getY());
    }
}
