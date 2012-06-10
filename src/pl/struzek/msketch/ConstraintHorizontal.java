package pl.struzek.msketch;

public class ConstraintHorizontal extends ConstraintLinesPerpendicular {

	public ConstraintHorizontal (Point K, Point L) {
		super(K, L, FixLine.Y.a,FixLine.Y.b);
		this.constraintType = GeometricConstraintType.SetHorizontal;
	}

}