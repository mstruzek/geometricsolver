package com.mstruzek.graphic;

import java.awt.geom.Ellipse2D;
import java.awt.Point;





public class MyPoint extends Point{


	boolean Dragged = false;

	int id ;
	
	int difX,difY;

	boolean isMouseClicked = false;

	
	public MyPoint(int p) {
		id = p;
	}

	public MyPoint(int x, int y) {
		super(x,y);
	}

	public boolean contains(int ix, int iy,double d) {
		com.mstruzek.msketch.Point p = com.mstruzek.msketch.Point.getDbPoint().get(this.id);
		Ellipse2D.Double circle = new Ellipse2D.Double(p.getX()-d/2,p.getY()-d/2,d,d);
		return circle.contains(ix, iy);
	}
	
	public boolean isDragged() {
		return Dragged;
	}

	public void setDragged(boolean dragged) {
		Dragged = dragged;
	}

	public int getDifX() {
		return difX;
	}

	public void setDifX(int difX) {
		this.difX = difX;
	}

	public int getDifY() {
		return difY;
	}

	public void setDifY(int difY) {
		this.difY = difY;
	}


	@Override
	public double getX() {
		return com.mstruzek.msketch.Point.getDbPoint().get(this.id).getX();
	}

	@Override
	public double getY() {
		return com.mstruzek.msketch.Point.getDbPoint().get(this.id).getY();
	}

	@Override
	public void setLocation(double x, double y) {
		com.mstruzek.msketch.Point p = com.mstruzek.msketch.Point.getDbPoint().get(this.id);
		p.setX(x);
		p.setY(y);	
	}

	@Override
	public void setLocation(Point ps) {
		super.setLocation(ps);
		com.mstruzek.msketch.Point p = com.mstruzek.msketch.Point.getDbPoint().get(this.id);
		p.setX(ps.getX());
		p.setY(ps.getY());	
	}

	public java.awt.Point getLocation() {
		return new java.awt.Point((int)getX(),(int)getY());
	}
}
