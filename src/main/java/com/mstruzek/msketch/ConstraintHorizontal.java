package com.mstruzek.msketch;

public class ConstraintHorizontal extends ConstraintLinesPerpendicular {

    public ConstraintHorizontal(int constId, Point K, Point L) {
        super(constId, K, L, FixLine.Y.a, FixLine.Y.b);
        this.constraintType = ConstraintType.SetHorizontal;
    }

}