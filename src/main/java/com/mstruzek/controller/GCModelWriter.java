package com.mstruzek.controller;

import com.mstruzek.msketch.*;

import java.io.BufferedWriter;
import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.util.Objects;
import java.util.stream.Stream;

import static java.lang.String.format;

/**
 * Format Read/Write
 * 
 * Descriptor(Parameter) ID(10) VALUE(23.02);
 * Descriptor(Parameter) ID(10) VALUE(23.02);
 * Descriptor(Parameter) ID(10) VALUE(23.02);
 * Descriptor(Parameter) ID(10) VALUE(23.02);
 *
 * Descriptor(GeometricObject) ID(10)  TYPE(FreePoint) P1(10.02) P2(192.02) P3(232.32) Param(1);
 * Descriptor(GeometricObject) ID(10)  TYPE(FreePoint) P1(10.02) P2(192.02) P3(232.32) Param(1);
 *
 * Descriptor(Constraint) ID(101) TYPE(HORIZONTAL) K(1) L(2) M(3) N(4);
 * Descriptor(Constraint) ID(101) TYPE(HORIZONTAL) K(1) L(2) M(3) N(4);
 * Descriptor(Constraint) ID(101) TYPE(HORIZONTAL) K(1) L(2) M(3) N(4);
 * Descriptor(Constraint) ID(101) TYPE(HORIZONTAL) K(1) L(2) M(3) N(4);
 *
 */
public class GCModelWriter implements Closeable {

    private BufferedWriter buff;

    public GCModelWriter(File filePath) {
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
        buff.write("#--signature: GeometricConstraintSolver  2009-2022\n");
        buff.write("#--file-format: V1\n");
        buff.write("\n");
        buff.write("#--definition-begin: \n");
        buff.write("\n");
    }

    public void writePoints() throws IOException {
        for (GeometricObject geometricObject : ModelRegistry.dbPrimitives.values()) {
            Point P1 = ModelRegistry.dbPoint.getOrDefault(geometricObject.getP1(), Point.EMPTY);
            Point P2 = ModelRegistry.dbPoint.getOrDefault(geometricObject.getP2(), Point.EMPTY);
            Point P3 = ModelRegistry.dbPoint.getOrDefault(geometricObject.getP3(), Point.EMPTY);
            for (Point point : Stream.of(P1, P2, P3).filter(point -> !Objects.equals(point, Point.EMPTY)).toList()) {
                buff.write("Descriptor(Point)");
                buff.write(format("%-9s", " ID(" + ObjectSerializer.writeToString(point.getId()) + ")"));
                buff.write(" PX(" + ObjectSerializer.writeToString(point.getX()) + ")");
                buff.write(" PY(" + ObjectSerializer.writeToString(point.getY()) + ")");
                buff.write(";\n"); /// Descriptor-END
            }
        }
        buff.write("\n");
    }

    public void writeParameters() throws IOException {
        for (Parameter parameter : ModelRegistry.dbParameter().values()) {
            buff.write("Descriptor(Parameter)");
            buff.write(format("%-9s", " ID(" + ObjectSerializer.writeToString(parameter.getId()) + ")"));
            buff.write(" VALUE(" + ObjectSerializer.writeToString(parameter.getValue()) + ")");
            buff.write(";\n"); /// Descriptor-END
        }
        buff.write("\n");
    }

    public void writeGeometricObjects() throws IOException {
        for (GeometricObject geometricObject : ModelRegistry.dbPrimitives.values()) {
            final int gpID = geometricObject.getPrimitiveId();
            if (gpID >= 0) {
                GeometricType primitiveType = geometricObject.getType();
                int P1 = geometricObject.getP1();
                int P2 = geometricObject.getP2();
                int P3 = geometricObject.getP3();
                buff.write("Descriptor(GeometricObject)");
                buff.write(format("%-9s"," ID(" + ObjectSerializer.writeToString(gpID) + ")"));
                buff.write(format("%-15s", " TYPE(" + ObjectSerializer.writeToString(primitiveType) + ")"));
                buff.write(format("%-9s", " P1(" + ObjectSerializer.writeToString(P1) + ")"));
                buff.write(format("%-9s", " P2(" + ObjectSerializer.writeToString(P2) + ")"));
                buff.write(" P3(" + ObjectSerializer.writeToString(P3) + ")");
                buff.write(";\n"); /// Descriptor-END
            }
        }
        buff.write("\n");
    }

    public void writeConstraints() throws IOException {
        for (Constraint constraint : ModelRegistry.dbConstraint.values().stream().filter(Constraint::isPersistent).toList()) {
            int cID = constraint.getConstraintId();
            ConstraintType constraintType = constraint.getConstraintType();
            int K = constraint.getK();
            int L = constraint.getL();
            int M = constraint.getM();
            int N = constraint.getN();
            int PARAM = constraint.getParameter();
            buff.write("Descriptor(Constraint)");
            buff.write(format("%-9s",  " ID(" + ObjectSerializer.writeToString(cID) + ")"));
            buff.write(format("%-25s",  " TYPE(" + ObjectSerializer.writeToString(constraintType) + ")"));
            buff.write(format("%-9s", " K(" + ObjectSerializer.writeToString(K) + ")"));
            buff.write(format("%-9s", " L(" + ObjectSerializer.writeToString(L) + ")"));
            buff.write(format("%-9s", " M(" + ObjectSerializer.writeToString(M) + ")"));
            buff.write(format("%-9s", " N(" + ObjectSerializer.writeToString(N) + ")"));
            buff.write(format("%-9s",  " PARAM(" + ObjectSerializer.writeToString(PARAM) + ")"));
            buff.write(";\n"); /// Descriptor-END
        }
        buff.write("\n");
    }

    public void writeClose() throws IOException {
        buff.write("\n");
        buff.write("#--definition-end: \n");
        buff.write("\n");
    }

}
