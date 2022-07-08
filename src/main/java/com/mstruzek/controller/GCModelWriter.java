package com.mstruzek.controller;

import com.mstruzek.msketch.*;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Stream;

import static java.util.stream.Collectors.toList;
import static java.util.stream.Collectors.toSet;

/**
 * Format Read/Write
 * <p>
 * BEGIN Parameter:
 * ID: 10;
 * VALUE: 23.02;
 * END;
 * <p>
 * BEGIN GeometricPrimitive:
 * ID: 10;
 * TYPE: FreePoint;
 * P1: 10.02;
 * P2: 192.02;
 * P3: 232.32;
 * Param: 1;
 * END;
 * <p>
 * BEGIN Constraint:
 * ID: 101;
 * TYPE: HORIZONTAL
 * K: 1;
 * L: 2;
 * M: 3;
 * N: 4;
 * END;
 */

public class GCModelWriter implements Closeable {

    private BufferedWriter buff;

    public GCModelWriter(File filePath) {
        try {
            buff = Files.newBufferedWriter(filePath.toPath(), StandardOpenOption.WRITE, StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING);
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void close() {
        if(buff != null) {
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
        for(GeometricPrimitive geometricPrimitive : GeometricPrimitive.dbPrimitives.values()) {
            Point P1 = Point.getDbPoint().getOrDefault(geometricPrimitive.getP1(), Point.EMPTY);
            Point P2 = Point.getDbPoint().getOrDefault(geometricPrimitive.getP2(), Point.EMPTY);
            Point P3 = Point.getDbPoint().getOrDefault(geometricPrimitive.getP3(), Point.EMPTY);
            for(Point point : Stream.of(P1, P2, P3).filter(point -> !Objects.equals(point, Point.EMPTY)).collect(toList())) {
                buff.write("BEGIN Point:\n");
                buff.write("     ID: " + Integer.toString(point.getId()) + ";\n");
                buff.write("     PX: " + Double.toString(point.getX()) + ";\n");
                buff.write("     PY: " + Double.toString(point.getY()) + ";\n");
                buff.write("END;\n");
                buff.write("\n");
            }
        }
        buff.write("\n");
    }

    public void writeParameters() throws IOException {
        for(Parameter parameter : Parameter.dbParameter.values()) {
            buff.write("BEGIN Parameter:\n");
            buff.write("     ID: " + Integer.toString(parameter.getId()) + ";\n");
            buff.write("     VALUE: " + Double.toString(parameter.getValue()) + ";\n");
            buff.write("END;\n");
            buff.write("\n");
        }
        buff.write("\n");
    }

    public void writeGeometricPrimitives() throws IOException {
        for(GeometricPrimitive geometricPrimitive : GeometricPrimitive.dbPrimitives.values()) {
            final int gpID = geometricPrimitive.getPrimitiveId();
            if(gpID >= 0) {
                GeometricPrimitiveType primitiveType = geometricPrimitive.getType();
                int P1 = geometricPrimitive.getP1();
                int P2 = geometricPrimitive.getP2();
                int P3 = geometricPrimitive.getP3();
                buff.write("BEGIN GeometricPrimitive:\n");
                buff.write("     ID: " + Integer.toString(gpID) + ";\n");
                buff.write("     TYPE: " + primitiveType.name() + ";\n");
                buff.write("     P1: " + Integer.toString(P1) + ";\n");
                buff.write("     P2: " + Integer.toString(P2) + ";\n");
                buff.write("     P3: " + Integer.toString(P3) + ";\n");
                buff.write("END;\n");
            }
            buff.write("\n");
        }
        buff.write("\n");
    }

    public void writeConstraints() throws IOException {
        Set<Integer> associate = GeometricPrimitive.dbPrimitives.values().stream()
            .flatMap(primitive -> Arrays.stream(primitive.associateConstraintsId()).boxed())
            .collect(toSet());
        for(Constraint constraint : Constraint.dbConstraint.values().stream().filter(c -> !associate.contains(c.getConstraintId())).collect(toList())) {
            int cID = constraint.getConstraintId();
            GeometricConstraintType constraintType = constraint.getConstraintType();
            int K = constraint.getK();
            int L = constraint.getL();
            int M = constraint.getM();
            int N = constraint.getN();
            int PARAM = constraint.getParametr();
            buff.write("BEGIN Constraint:\n");
            buff.write("     ID: " + Integer.toString(cID) + ";\n");
            buff.write("     TYPE: " + constraintType.name() + ";\n");
            buff.write("     K: " + Integer.toString(K) + ";\n");
            buff.write("     L: " + Integer.toString(L) + ";\n");
            buff.write("     M: " + Integer.toString(M) + ";\n");
            buff.write("     N: " + Integer.toString(N) + ";\n");
            buff.write("     PARAM: " + Integer.toString(PARAM) + ";\n");
            buff.write("END;\n");
            buff.write("\n");
        }
        buff.write("\n");
    }

    public void writeClose() throws IOException {
        buff.write("\n");
        buff.write("#--definition-end: \n");
        buff.write("\n");
    }

}
