package com.mstruzek.msketch;

class BoundingBox {

    double left;
    double bottom;
    double width;
    double height;

    public BoundingBox fillInPoint(Point point) {
        left = Math.min(point.x, left);
        width = Math.max(point.x, width);
        bottom = Math.min(point.y, bottom);
        height = Math.max(point.y, height);
        return this;
    }

    public BoundingBox fillInBoundingBox(BoundingBox boundingBox) {
        left = Math.min(boundingBox.left, left);
        width = Math.max(boundingBox.width, width);
        bottom = Math.min(boundingBox.bottom, bottom);
        height = Math.max(boundingBox.height, height);
        return null;
    }

    public double length() {
        return Math.sqrt((left-width)*(left-width) + (height-bottom)* (height- bottom));
    }
}
