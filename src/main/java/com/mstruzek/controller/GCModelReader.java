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

    private static final String HEADER_PREFIX = "#";
    private static final int START_PREFIX = 0;
    private static final Pattern HEADER_FIELD_PATTERN = Pattern.compile("^#(--.*): (.*)$");

    private static final String FILE_FORMAT_VERSION = "V1";
    private static final String HEADER_SIGNATURE = "--signature";
    private static final String HEADER_FILE_FORMAT = "--file-format";
    private static final String HEADER_DEFINITION_BEGIN = "--definition-begin";
    private static final String HEADER_DEFINITION_END = "--definition-end";

    private static final int DESCRIPTOR_TYPE_OFFSET = 0;
    private static final String DESCRIPTOR_POINT = "Descriptor(Point)";
    private static final String DESCRIPTOR_GEOMETRIC_PRIMITIVE = "Descriptor(GeometricPrimitive)";
    private static final String DESCRIPTOR_PARAMETER = "Descriptor(Parameter)";
    private static final String DESCRIPTOR_CONSTRAINT = "Descriptor(Constraint)";

    private BufferedReader buff;

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
        String input = null;
        String[] fieldValue;

        while ((input = buff.readLine()) != null) {

            /// skip empty lines
            if (input.isBlank())
                continue;

            if (input.indexOf(HEADER_PREFIX) == START_PREFIX) {
                /// Comments - header data
                processComments(input);
                continue;
            }

            fieldValue = input.split("\\s+");
            switch (fieldValue[DESCRIPTOR_TYPE_OFFSET]) {
                case DESCRIPTOR_POINT ->                processDescriptorPoint(fieldValue);
                case DESCRIPTOR_GEOMETRIC_PRIMITIVE ->  processDescriptorPrimitive(fieldValue);
                case DESCRIPTOR_PARAMETER ->            processDescriptorParameter(fieldValue);
                case DESCRIPTOR_CONSTRAINT ->           processDescriptorConstraint(fieldValue);
                default -> {
                    throw new Error("unrecognized object " + input);
                }
            }
        }
    }

    private static void processDescriptorParameter(String[] fieldValue) {

        final int parameterId = ObjectDeserializer.toInteger(processField(fieldValue[1]));
        final double parameterValue = ObjectDeserializer.toDouble(processField(fieldValue[2]));

        /// store in db
        Parameter parameter;

        parameter = new Parameter(parameterId, parameterValue);
        ModelRegistry.registerParameter(parameterId, parameter);
    }

    private static void processDescriptorPoint(String[] fieldValue) {

        final int pointId = ObjectDeserializer.toInteger(processField(fieldValue[1]));
        final double coordinateX = ObjectDeserializer.toDouble(processField(fieldValue[2]));
        final double coordinateY = ObjectDeserializer.toDouble(processField(fieldValue[3]));

        /// save point
        Point point;

        point = new Point(pointId, coordinateX, coordinateY);
        ModelRegistry.registerPoint(pointId, point);
    }

    /// @@ bindings primitive types to domain model !
    private static void processDescriptorPrimitive(String[] fieldValue) {

        int primitiveId = ObjectDeserializer.toInteger(processField(fieldValue[1]));
        GeometricPrimitiveType primitiveType = Enum.valueOf(GeometricPrimitiveType.class, processField(fieldValue[2]));
        int p1 = ObjectDeserializer.toInteger(processField(fieldValue[3]));
        int p2 = ObjectDeserializer.toInteger(processField(fieldValue[4]));
        int p3 = ObjectDeserializer.toInteger(processField(fieldValue[5]));

        Point P1 = null;
        Point P2 = null;
        Point P3 = null;

        if (p1 != -1) P1 = ModelRegistry.dbPoint.get(p1);
        if (p2 != -1) P2 = ModelRegistry.dbPoint.get(p2);
        if (p3 != -1) P3 = ModelRegistry.dbPoint.get(p3);

        /// save point

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
            default:
                throw new Error("invalid input fieldLine : " + Arrays.toString(fieldValue));
        }

    }

    private static final Pattern FIELD_VALUE = Pattern.compile("^\\w*?\\((.*?)\\);?$");

    public static String processField(String input) {
        Matcher m = FIELD_VALUE.matcher(input);
        if (!m.find()) {
            throw new Error("failed line matcher " + input);
        } else {
            return m.group(1);
        }
    }

    public static void processDescriptorConstraint(String[] fieldValue) {

        int constId = ObjectDeserializer.toInteger(processField(fieldValue[1]));
        GeometricConstraintType constraintType = Enum.valueOf(GeometricConstraintType.class, processField(fieldValue[2]));
        int vK = ObjectDeserializer.toInteger(processField(fieldValue[3]));
        int vL = ObjectDeserializer.toInteger(processField(fieldValue[4]));
        int vM = ObjectDeserializer.toInteger(processField(fieldValue[5]));
        int vN = ObjectDeserializer.toInteger(processField(fieldValue[6]));
        int paramId = ObjectDeserializer.toInteger(processField(fieldValue[7]));

        /// save point
        Point K = null;
        Point L = null;
        Point M = null;
        Point N = null;
        Parameter parameter = null;

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
                throw new Error("invalid fieldValue fieldLine : " + Arrays.toString(fieldValue));
        }
    }

    /**
     * Skip all comments , and other markers but remember to check file format version.
     */
    private void processComments(String inputLine) {

        Matcher matcher = HEADER_FIELD_PATTERN.matcher(inputLine);
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
                    throw new Error("unsupported header input  ! =  `" + inputLine + "`");
            }
        } else {
            throw new Error("unrecognized input ! =  `" + inputLine + "`");
        }
    }

}
