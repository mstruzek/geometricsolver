package com.mstruzek.msketch;

public class ConstraintVertical extends ConstraintLinesPerpendicular{

    public ConstraintVertical(int constId,Point K,Point L){
        super(constId,K,L,FixLine.X.a,FixLine.X.b);
        this.constraintType=GeometricConstraintType.SetVertical;
    }

}
