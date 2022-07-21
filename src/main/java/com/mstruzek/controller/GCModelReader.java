package com.mstruzek.controller;

import com.mstruzek.msketch.*;

import java.io.BufferedReader;
import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class GCModelReader implements Closeable {

    private static final Pattern HEADER_FIELD_PATTERN = Pattern.compile("^#(--.*): (.*)$");

    private static final String FILE_FORMAT_VERSION = "V1";
    private static final String HEADER_SIGNATURE = "--signature";
    private static final String HEADER_FILE_FORMAT = "--file-format";
    private static final String HEADER_DEFINITION_BEGIN = "--definition-begin";
    private static final String HEADER_DEFINITION_END = "--definition-end";

    static final Pattern STRUCT_FIELD_PATTERN = Pattern.compile("^\\s{5}(.*): (.*);$");

    private static final String BEGIN_POINT = "BEGIN Point:";
    private static final String BEGIN_PRIMITIVE = "BEGIN GeometricPrimitive:";
    private static final String BEGIN_CONSTRAINT = "BEGIN Constraint:";
    private static final String BEGIN_PARAMETER = "BEGIN Parameter:";

    private static final String PATTERN_END = "END;";

    private static final int STA_END = -1;
    private static final int STA_POINT = 0;
    private static final int STA_PRIMITIVE = 1;
    private static final int STA_CONSTRAINT = 2;
    private static final int STA_PARAMETER = 3;

    private int lstate = STA_END;
    private String input = null;
    private BufferedReader buff;

    private Object[] slots = new Object[]{
        null, //  0: ID
        null, //  1: PX
        null, //  2: PY
        null, //  3: TYPE
        null, //  4: P1
        null, //  5: P2
        null, //  6: P3
        null, //  7: K
        null, //  8: L
        null, //  9: M
        null, // 10: N
        null, // 11: PARAM
        null, // 12: VALUE
    };

    public GCModelReader(File filePath) {
        try {
            buff = Files.newBufferedReader(filePath.toPath());
        } catch (IOException e) {
            throw new Error(e);
        }
    }

    @Override
    public void close() throws IOException {
        if (buff != null) {
            buff.close();
            buff = null;
        }
    }


    public void readModel() throws IOException {
        input = null;
        lstate = STA_END;
        while ((input = buff.readLine()) != null) {
            switch (lstate) {
                case STA_END:
                    processEnd();
                    break;
                case STA_POINT:
                    processPoint();
                    break;
                case STA_PRIMITIVE:
                    processPrimitive();
                    break;
                case STA_CONSTRAINT:
                    processConstraint();
                    break;
                case STA_PARAMETER:
                    processParameter();
                    break;
                default:
                    throw new Error("lexer state not recognized: " + lstate);
            }
        }
    }

    private void processParameter() {
        /**
         * BEGIN Parameter:
         *      ID: 0;
         *      VALUE: 10000.0;
         * END;
         */

        Matcher matcher = STRUCT_FIELD_PATTERN.matcher(input);
        //@@@ - wiazanki
        if (matcher.matches()) {
            String fieldName = matcher.group(1);
            String fieldValue = matcher.group(2);
            switch (fieldName) {
                case "ID":
                    slots[0] = ObjectDeserializer.toInteger(fieldValue);
                    return;
                case "VALUE":
                    slots[12] = ObjectDeserializer.toDouble(fieldValue);
                    return;
                default:
                    throw new Error("invalid input fieldLine : " + input);
            }
        }

        if (PATTERN_END.equals(input)) {
            Integer parameterId = (Integer) slots[0];
            Double parameterValue = (Double) slots[12];
            /// store in db
            Parameter parameter = new Parameter(parameterId, parameterValue);

            ModelRegistry.registerParameter(parameterId, parameter);

            Arrays.fill(slots, 0, slots.length, null);
            lstate = STA_END;
        } else {
            throw new Error("invalid input fieldLine : " + input);
        }

    }

    private void processPoint() {
        /*
         *      ID: 1;
         *      PX: 189.383457013559;
         *      PY: 138.6485791055082;
         */
        Matcher matcher = STRUCT_FIELD_PATTERN.matcher(input);
        //@@@ - wiazanki
        if (matcher.matches()) {
            String fieldName = matcher.group(1);
            String fieldValue = matcher.group(2);
            switch (fieldName) {
                case "ID":
                    slots[0] = ObjectDeserializer.toInteger(fieldValue);
                    return;
                case "PX":
                    slots[1] = ObjectDeserializer.toDouble(fieldValue);
                    return;
                case "PY":
                    slots[2] = ObjectDeserializer.toDouble(fieldValue);
                    return;
                default:
                    throw new Error("invalid input fieldLine : " + input);
            }
        }

        if (PATTERN_END.equals(input)) {
            Integer pointId = (Integer) slots[0];
            Double coordinateX = (Double) slots[1];
            Double coordinateY = (Double) slots[2];
            /// save point
            Point point = new Point(pointId, coordinateX, coordinateY);

            ModelRegistry.registerPoint(pointId, point);

            Arrays.fill(slots, 0, slots.length, null);
            lstate = STA_END;
        } else {
            throw new Error("invalid input fieldLine : " + input);
        }
    }


    /// @@ bindings primitive types to domain model !
    private void processPrimitive() {
        /*
         *      ID: 1;
         *      TYPE: Line;
         *      P1: 5;
         *      P2: 6;
         *      P3: -1;
         */
        Matcher matcher = STRUCT_FIELD_PATTERN.matcher(input);
        if (matcher.matches()) {
            String fieldName = matcher.group(1);
            String fieldValue = matcher.group(2);
            switch (fieldName) {
                case "ID":
                    slots[0] = ObjectDeserializer.toInteger(fieldValue);
                    return;
                case "TYPE":
                    slots[3] = ObjectDeserializer.toString(fieldValue);
                    return;
                case "P1":
                    slots[4] = ObjectDeserializer.toInteger(fieldValue);
                    return;
                case "P2":
                    slots[5] = ObjectDeserializer.toInteger(fieldValue);
                    return;
                case "P3":
                    slots[6] = ObjectDeserializer.toInteger(fieldValue);
                    return;
                default:
                    throw new Error("invalid input fieldLine : " + input);
            }
        }

        if (PATTERN_END.equals(input)) {
            /// save point
            int primitiveId;
            GeometricPrimitiveType primitiveType;
            int p1;
            int p2;
            int p3;

            Point P1 = null;
            Point P2 = null;
            Point P3 = null;

            primitiveType = Enum.valueOf(GeometricPrimitiveType.class, (String) slots[3]);
            primitiveId = (Integer) slots[0];
            p1 = (Integer) slots[4];
            p2 = (Integer) slots[5];
            p3 = (Integer) slots[6];


            if (p1 != -1) P1 = ModelRegistry.dbPoint.get(p1);
            if (p2 != -1) P2 = ModelRegistry.dbPoint.get(p2);
            if (p3 != -1) P3 = ModelRegistry.dbPoint.get(p3);

            switch (primitiveType) {
                case FreePoint:
                    Model.addPoint(primitiveId, (P1));
                    break;
                case FixLine:
                    throw new Error("axis OXY is not serialized");
                case Line:
                    Model.addLine(primitiveId, (P1), (P2));
                    break;
                case Circle:
                    Model.addCircle(primitiveId, (P1), (P2));
                    break;
                case Arc:
                    Model.addArc(primitiveId, (P1), (P2), (P3));
                    break;
            }
            Arrays.fill(slots, 0, slots.length, null);
            lstate = STA_END;
        } else {
            throw new Error("invalid input fieldLine : " + input);
        }
    }

    private void processConstraint() {
        /*
         *      ID: 4;
         *      TYPE: Connect2Points;
         *      K: 2;
         *      L: 5;
         *      M: -1;
         *      N: -1;
         *      PARAM: -1;
         */
        Matcher matcher = STRUCT_FIELD_PATTERN.matcher(input);
        if (matcher.matches()) {
            String fieldName = matcher.group(1);
            String fieldValue = matcher.group(2);
            switch (fieldName) {
                case "ID":
                    slots[0] = ObjectDeserializer.toInteger(fieldValue);
                    return;
                case "TYPE":
                    slots[3] = ObjectDeserializer.toString(fieldValue);
                    return;
                case "K":
                    slots[7] = ObjectDeserializer.toInteger(fieldValue);
                    return;
                case "L":
                    slots[8] = ObjectDeserializer.toInteger(fieldValue);
                    return;
                case "M":
                    slots[9] = ObjectDeserializer.toInteger(fieldValue);
                    return;
                case "N":
                    slots[10] = ObjectDeserializer.toInteger(fieldValue);
                    return;
                case "PARAM":
                    slots[11] = ObjectDeserializer.toInteger(fieldValue);
                    return;
                default:
                    throw new Error("invalid input fieldLine : " + input);
            }
        }

        if (PATTERN_END.equals(input)) {
            /// save point
            int constId;
            GeometricConstraintType constraintType;
            int vK;
            int vL;
            int vM;
            int vN;
            int paramId;

            Point K = null;
            Point L = null;
            Point M = null;
            Point N = null;
            Parameter parameter = null;

            constraintType = Enum.valueOf(GeometricConstraintType.class, (String) slots[3]);
            constId = (Integer) slots[0];
            vK = (Integer) slots[7];
            vL = (Integer) slots[8];
            vM = (Integer) slots[9];
            vN = (Integer) slots[10];
            paramId = (Integer) slots[11];

            if (vK != -1) K = ModelRegistry.dbPoint.get(vK);
            if (vL != -1) L = ModelRegistry.dbPoint.get(vL);
            if (vM != -1) M = ModelRegistry.dbPoint.get(vM);
            if (vN != -1) N = ModelRegistry.dbPoint.get(vN);
            if (paramId != -1 && constraintType.isParametrized()) {
                parameter = ModelRegistry.dbParameter.get(paramId);
            }

            switch (constraintType) {
                case FixPoint:
                case ParametrizedXFix:
                case ParametrizedYFix:
                case Connect2Points:
                case Distance2Points:
                case DistancePointLine:
                case LinesParallelism:
                case LinesPerpendicular:
                case Angle2Lines:
                case SetHorizontal:
                case SetVertical:
                case HorizontalPoint:
                case VerticalPoint:
                case EqualLength:
                case ParametrizedLength:
                case Tangency:
                case CircleTangency:
                    Model.addConstraint(constId, constraintType, K, L, M, N, parameter);
                    break;
                default:
                    throw new Error("invalid input fieldLine : " + input);
            }
            Arrays.fill(slots, 0, slots.length, null);
            lstate = STA_END;
        } else {
            throw new Error("invalid input fieldLine : " + input);
        }
    }

    /**
     * Skip all comments , and other markers but remember to check file format version.
     */
    private void processEnd() {

        switch (input) {
            case "":
                // continue on empty line
                return;
            case BEGIN_POINT:
                lstate = STA_POINT;
                return;
            case BEGIN_PRIMITIVE:
                lstate = STA_PRIMITIVE;
                return;
            case BEGIN_CONSTRAINT:
                lstate = STA_CONSTRAINT;
                return;
            case BEGIN_PARAMETER:
                lstate = STA_PARAMETER;
                return;
        }

        Matcher matcher = HEADER_FIELD_PATTERN.matcher(input);
        if (matcher.matches()) {
            String fieldName = matcher.group(1);
            String fieldValue = matcher.group(2);
            switch (fieldName) {
                case HEADER_SIGNATURE:
                    Reporter.notify("model descriptor  signature =>  ` " + fieldValue + " `");
                case HEADER_DEFINITION_BEGIN:
                    Reporter.notify("model descriptor  =>  begin");
                    break;
                case HEADER_DEFINITION_END:
                    Reporter.notify("model descriptor  =>  end");
                    break;
                case HEADER_FILE_FORMAT:
                    Reporter.notify("model descriptor version  : " + fieldValue);
                    if (!FILE_FORMAT_VERSION.equals(fieldValue))
                        throw new Error("unsupported format version ! =  " + fieldValue);
                    break;
                default:
                    throw new Error("unsupported header input  ! =  `" + input + "`");
            }
        } else {
            throw new Error("unrecognized input ! =  `" + input + "`");
        }
    }

}
