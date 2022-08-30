package com.mstruzek.controller;

import com.mstruzek.msketch.*;

import java.io.BufferedWriter;
import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.util.Locale;

import static java.lang.String.format;

/**
 * Provides a way to write model into declarative  c++ code
 *
 * err = jni_registerPointType(&env, eclass, 4, 20.0, 0.0);
 * err = jni_registerPointType(&env, eclass, 5, 20.0, 210.0);
 * err = jni_registerPointType(&env, eclass, 6, 20.0, 600.0);
 * err = jni_registerPointType(&env, eclass, 7, 20.0, 800.0);
 * err = jni_registerConstraintType(&env, eclass, 3, CONSTRAINT_TYPE_ID_FIX_POINT, 4, -1, -1, -1, -1, 20.0, 0.0);
 * err = jni_registerConstraintType(&env, eclass, 4, CONSTRAINT_TYPE_ID_FIX_POINT, 7, -1, -1, -1, -1, 20.0, 800.0);
 * err = jni_registerGeometricType(&env, eclass, 2, GEOMETRIC_TYPE_ID_LINE, 5, 6, 4, 7, -1, -1, -1);
 *
 * /// constraint(1,5)(Connect2Points)
 * err = jni_registerConstraintType(&env, eclass, 5, CONSTRAINT_TYPE_ID_CONNECT_2_POINTS, 1, 5, -1, -1, -1, -1, -1);
 */
public class CppModelWriter implements Closeable {

    private BufferedWriter buff;

    private static Locale l = Locale.ROOT;

    private File filePath;

    public CppModelWriter(File filePath) {
        this.filePath =filePath;
        try {
            buff = Files.newBufferedWriter(filePath.toPath(), StandardOpenOption.WRITE, StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void close() {
        if (buff != null) {
            try {
                buff.flush();
                buff.close();
                buff = null;
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    public void writeHeader() throws IOException {
        buff.write("/// --signature: GeometricConstraintSolver  2009-2022\n");
        buff.write("/// --file-format: V1\n");
        buff.write(format("/// --file-name: %s \n", filePath.getAbsolutePath()));
        buff.write("\n");
        buff.write("/// --definition-begin: \n");
        buff.write("\n");
    }

    public void writePoints() throws IOException {
        for (GeometricObject geometricObject : ModelRegistry.dbPrimitives.values()) {
            for (Point point : geometricObject.getAllPoints()) {
                buff.write(format(l, "err = jni_registerPointType(&env, eclass, %d, %e, %e); \n", point.getId(), point.getX(), point.getY()));
            }
        }
        buff.write("\n");
    }

    public void writeParameters() throws IOException {
        for (Parameter parameter : ModelRegistry.dbParameter().values()) {
            buff.write(format(l, "err = jni_registerParameterType(&env, eclass, %d, %e); \n", parameter.getId(), parameter.getValue()));
        }
        buff.write("\n");
    }

    public void writeGeometricObjects() throws IOException {
        for (GeometricObject geometricObject : ModelRegistry.dbPrimitives.values()) {
            final int gpID = geometricObject.getPrimitiveId();

            if (gpID >= 0) {
                GeometricType primitiveType = geometricObject.getType();
                String geometricType = geometricTypeName(primitiveType);
                int p1 = geometricObject.getP1();
                int p2 = geometricObject.getP2();
                int p3 = geometricObject.getP3();
                int pA = geometricObject.getA();
                int pB = geometricObject.getB();
                int pC = geometricObject.getC();
                int pD = geometricObject.getD();
                buff.write(format(l, "err = jni_registerGeometricType(&env, eclass, %d, %s, %d, %d, %d, %d, %d, %d, %d); \n", gpID, geometricType, p1, p2, p3, pA, pB, pC, pD));
            }
        }
        buff.write("\n");
    }

    private String geometricTypeName(GeometricType primitiveType) {
        return switch (primitiveType) {
            case FreePoint -> "GEOMETRIC_TYPE_ID_FREE_POINT";
            case Line -> "GEOMETRIC_TYPE_ID_LINE";
            case Circle -> "GEOMETRIC_TYPE_ID_CIRCLE";
            case FixLine -> "GEOMETRIC_TYPE_ID_FIX_LINE";
            case Arc -> "GEOMETRIC_TYPE_ID_ARC";
        };
    }

    public void writeConstraints() throws IOException {
        for (Constraint constraint : ModelRegistry.dbConstraint.values()) {
            int cID = constraint.getConstraintId();
            ConstraintType constraintType = constraint.getConstraintType();
            String constraintName = constraintTypeName(constraintType);
            int K = constraint.getK();
            int L = constraint.getL();
            int M = constraint.getM();
            int N = constraint.getN();
            int PARAM = constraint.getParameter();
            double vecX = 0.0;
            double vecY = 0.0;
            if (constraintType == ConstraintType.FixPoint) {
                vecX = ((ConstraintFixPoint) constraint).getFixVector().getX();
                vecY = ((ConstraintFixPoint) constraint).getFixVector().getY();
            }
            buff.write(format(l, "err = jni_registerConstraintType(&env, eclass, %d, %s, %d, %d, %d, %d, %d, %e, %e);\n", cID, constraintName, K, L, M, N, PARAM, vecX, vecY));
        }
        buff.write("\n");
    }

    private String constraintTypeName(ConstraintType constraintType) {
        return switch (constraintType) {
            case FixPoint -> "CONSTRAINT_TYPE_ID_FIX_POINT";
            case ParametrizedXFix -> "CONSTRAINT_TYPE_ID_PARAMETRIZED_XFIX";
            case ParametrizedYFix -> "CONSTRAINT_TYPE_ID_PARAMETRIZED_YFIX";
            case Connect2Points -> "CONSTRAINT_TYPE_ID_CONNECT_2_POINTS";
            case HorizontalPoint -> "CONSTRAINT_TYPE_ID_HORIZONTAL_POINT";
            case VerticalPoint -> "CONSTRAINT_TYPE_ID_VERTICAL_POINT";
            case LinesParallelism -> "CONSTRAINT_TYPE_ID_LINES_PARALLELISM";
            case LinesPerpendicular -> "CONSTRAINT_TYPE_ID_LINES_PERPENDICULAR";
            case EqualLength -> "CONSTRAINT_TYPE_ID_EQUAL_LENGTH";
            case ParametrizedLength -> "CONSTRAINT_TYPE_ID_PARAMETRIZED_LENGTH";
            case Tangency -> "CONSTRAINT_TYPE_ID_TANGENCY";
            case CircleTangency -> "CONSTRAINT_TYPE_ID_CIRCLE_TANGENCY";
            case Distance2Points -> "CONSTRAINT_TYPE_ID_DISTANCE_2_POINTS";
            case DistancePointLine -> "CONSTRAINT_TYPE_ID_DISTANCE_POINT_LINE";
            case Angle2Lines -> "CONSTRAINT_TYPE_ID_ANGLE_2_LINES";
            case SetHorizontal -> "CONSTRAINT_TYPE_ID_SET_HORIZONTAL";
            case SetVertical -> "CONSTRAINT_TYPE_ID_SET_VERTICAL";
        };
    }

    public void writeClose() throws IOException {
        buff.write("\n");
        buff.write("/// --definition-end: \n");
        buff.write("\n");
    }

}
