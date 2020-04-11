package com.mstruzek.msketch;

public class ConstraintVertical extends ConstraintLinesPerpendicular {

	public ConstraintVertical(Point K, Point L) {
		super(K, L, FixLine.X.a,FixLine.X.b);
		this.constraintType = GeometricConstraintType.SetVertical;
	}

}
