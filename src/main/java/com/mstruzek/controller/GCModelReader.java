package com.mstruzek.controller;

import com.mstruzek.msketch.*;

import java.io.BufferedReader;
import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Objects;
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

    private static final String DESCRIPTOR_POINT = "Point";
    private static final String DESCRIPTOR_GEOMETRIC_PRIMITIVE = "GeometricObject";
    private static final String DESCRIPTOR_PARAMETER = "Parameter";
    private static final String DESCRIPTOR_CONSTRAINT = "Constraint";

    private final Pattern STRUCT_FIELD_VALUE = Pattern.compile("([a-zA-Z0-9]+)\\(([a-zA-Z0-9\\.\\-]+)\\)");

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
        String input;

        while ((input = buff.readLine()) != null) {
            /// skip empty lines
            if (input.isBlank())
                continue;

            if (input.indexOf(HEADER_PREFIX) == START_PREFIX) {
                /// Comments - header data
                processComments(input);
                continue;
            }

            final Matcher m = STRUCT_FIELD_VALUE.matcher(input);
            final String descriptorType = requiredField(m, "Descriptor");

            switch (descriptorType) {
                case DESCRIPTOR_POINT ->                processDescriptorPoint(m);
                case DESCRIPTOR_GEOMETRIC_PRIMITIVE ->  processDescriptorPrimitive(m);
                case DESCRIPTOR_PARAMETER ->            processDescriptorParameter(m);
                case DESCRIPTOR_CONSTRAINT ->           processDescriptorConstraint(m);
                default -> {
                    throw new Error("unrecognized object " + input);
                }
            }
        }
    }


    private static String requiredField(Matcher iterator, String fieldName) {
        if(iterator.find() && Objects.equals(iterator.group(1), fieldName)) {
            return iterator.group(2);
        } else {
            throw new Error("required field is " + fieldName);
        }
    }

    private static void processDescriptorParameter(Matcher iterator) {

        final int parameterId = ObjectDeserializer.toInteger(requiredField(iterator,"ID"));
        final double parameterValue = ObjectDeserializer.toDouble(requiredField(iterator,"VALUE"));

        /// store in db
        Parameter parameter;

        parameter = new Parameter(parameterId, parameterValue);
        ModelRegistry.registerParameter(parameterId, parameter);
    }

    private static void processDescriptorPoint(Matcher iterator) {

        final int pointId = ObjectDeserializer.toInteger(requiredField(iterator, "ID"));
        final double coordinateX = ObjectDeserializer.toDouble(requiredField(iterator, "PX"));
        final double coordinateY = ObjectDeserializer.toDouble(requiredField(iterator, "PY"));

        /// save point
        Point point;

        point = new Point(pointId, coordinateX, coordinateY);
        ModelRegistry.registerPoint(pointId, point);
    }

    /// @@ bindings primitive types to domain model !
    private static void processDescriptorPrimitive(Matcher iterator) {

        int primitiveId = ObjectDeserializer.toInteger(requiredField(iterator, "ID"));
        GeometricType primitiveType = Enum.valueOf(GeometricType.class, requiredField(iterator, "TYPE"));
        int p1 = ObjectDeserializer.toInteger(requiredField(iterator, "P1"));
        int p2 = ObjectDeserializer.toInteger(requiredField(iterator, "P2"));
        int p3 = ObjectDeserializer.toInteger(requiredField(iterator, "P3"));

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
                throw new Error("invalid input fieldLine : " + iterator.group());
        }

    }

    public static void processDescriptorConstraint(Matcher iterator) {

        int constId = ObjectDeserializer.toInteger(requiredField(iterator, "ID"));
        ConstraintType constraintType = Enum.valueOf(ConstraintType.class, requiredField(iterator, "TYPE"));
        int vK = ObjectDeserializer.toInteger(requiredField(iterator, "K"));
        int vL = ObjectDeserializer.toInteger(requiredField(iterator, "L"));
        int vM = ObjectDeserializer.toInteger(requiredField(iterator, "M"));
        int vN = ObjectDeserializer.toInteger(requiredField(iterator, "N"));
        int paramId = ObjectDeserializer.toInteger(requiredField(iterator, "PARAM"));

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
                throw new Error("invalid iterator fieldLine : " + iterator.group());
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
