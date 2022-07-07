package com.mstruzek.controller;

import com.mstruzek.msketch.GeometricPrimitiveType;
import com.mstruzek.msketch.Model;
import com.mstruzek.msketch.Point;

import java.io.*;
import java.nio.file.Files;
import java.util.regex.Pattern;

public class GCModelReader implements Closeable{

    private static final Pattern PREFIX_COMMENTS=Pattern.compile("^#!(.*)$");
    private static final Pattern VAR_FILE_FORMAT=Pattern.compile("^#--file-format: (.*);$");
    private static final String FILE_FORMAT_VERSION="V1";

    private static final Pattern MARKER_DEFINITION_BEGIN=Pattern.compile("^#--definition-begin$");
    private static final Pattern MARKER_DEFINITION_END=Pattern.compile("^#--definition-end$");


    //FIXME rewrite singel PAT_FIELD = Pattern.compile("^\\s{5}(.*): (.*);$"); ??

    private static final Pattern BEGIN_POINT=Pattern.compile("^BEGIN Point:$");
    private static final Pattern PAT_ID=Pattern.compile("^\\s{5}ID: (\\d+);$");
    private static final Pattern PAT_PX=Pattern.compile("^\\s{5}PX: ([\\d.]+);$");
    private static final Pattern PAT_PY=Pattern.compile("^\\s{5}PX: ([\\d.]+);$");

    private static final Pattern BEGIN_PRIMITIVE=Pattern.compile("^BEGIN GeometricPrimitive:$");
    private static final Pattern PAT_TYPE=Pattern.compile("^\\s{5}(.*): (.*);$");
    private static final Pattern PAT_P1=Pattern.compile("^\\s{5}P1: (\\d+);$");
    private static final Pattern PAT_P2=Pattern.compile("^\\s{5}P2: (\\d+);$");
    private static final Pattern PAT_P3=Pattern.compile("^\\s{5}P3: (\\d+);$");

    private static final Pattern BEGIN_CONSTRAINT=Pattern.compile("^BEGIN Constraint:$");
    private static final Pattern PAT_K=Pattern.compile("^\\s{5}K: (\\d+);$");
    private static final Pattern PAT_L=Pattern.compile("^\\s{5}L: (\\d+);$");
    private static final Pattern PAT_M=Pattern.compile("^\\s{5}M: (\\d+);$");
    private static final Pattern PAT_N=Pattern.compile("^\\s{5}N: (\\d+);$");
    private static final Pattern PAT_PARAM=Pattern.compile("^\\s{5}PARAM: (\\d+);$");

    private static final Pattern PAT_END=Pattern.compile("^END;$");

    private static final int STA_END=-1;
    private static final int STA_POINT=0;
    private static final int STA_PRIMITIVE=1;
    private static final int STA_CONSTRAINT=2;

    private int lstate=STA_END;
    private String input=null;
    private BufferedReader buff;

    private Object[] slots= new Object[]{
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
    };

    public GCModelReader(File filePath){
        try{
            buff=Files.newBufferedReader(filePath.toPath());
        }catch(IOException e){
            throw new Error(e);
        }
    }

    @Override
    public void close() throws IOException{
        if(buff!=null){
            buff.close();
            buff=null;
        }
    }


    public void readModel() throws IOException{
        input=null;
        lstate=STA_END;
        while((input=buff.readLine())!=null){
            switch(lstate){
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
                default:
                    throw new Error("lexer state not recognized: "+lstate);
            }
        }
    }



    private void processPoint(){
/*
 *      ID: 1;
 *      PX: 189.383457013559;
 *      PY: 138.6485791055082;
 */
        if(PAT_ID.matcher(input).matches()) {
            slots[0] = Integer.parseInt(PAT_ID.matcher(input).group(1));
            return;
        }

        if(PAT_PX.matcher(input).matches()) {
            slots[1] = Double.parseDouble(PAT_PX.matcher(input).group(1));
            return;
        }

        if(PAT_PY.matcher(input).matches()) {
            slots[2] = Double.parseDouble(PAT_PY.matcher(input).group(1));
            return;
        }

        if(PAT_END.matcher(input).matches()) {
            /// save point
            Integer pointId=(Integer) slots[0];
            Double coordinateX=(Double) slots[1];
            Double coordinateY=(Double) slots[2];
            Point point=new Point(pointId, coordinateX, coordinateY);
            lstate=STA_END;
            return;
        }

        throw new Error("invalid input line : " + input);
    }


    private void processPrimitive(){
/*
 *      ID: 1;
 *      TYPE: Line;
 *      P1: 5;
 *      P2: 6;
 *      P3: -1;
 */
        if(PAT_ID.matcher(input).matches()) {
            slots[0] = Integer.parseInt(PAT_ID.matcher(input).group(1));
            return;
        }

        if(PAT_TYPE.matcher(input).matches()) {
            slots[3] = (PAT_TYPE.matcher(input).group(1));
            return;
        }

        if(PAT_P1.matcher(input).matches()) {
            slots[4] = Integer.parseInt(PAT_P1.matcher(input).group(1));
            return;
        }

        if(PAT_P2.matcher(input).matches()) {
            slots[5] = Integer.parseInt(PAT_P2.matcher(input).group(1));
            return;
        }

        if(PAT_P3.matcher(input).matches()) {
            slots[6] = Integer.parseInt(PAT_P3.matcher(input).group(1));
            return;
        }

        if(PAT_END.matcher(input).matches()) {
            /// CONSUME slots
            int primitiveId = (Integer) slots[0];
            int P1;
            int P2;
            int P3;
            GeometricPrimitiveType primitiveType=Enum.valueOf(GeometricPrimitiveType.class,(String) slots[3]);
            switch(primitiveType) {
                case FreePoint:
                    P1=(Integer)slots[4];
                    Model.addPoint(primitiveId, Point.dbPoint.get(P1));
                    break;
                case FixLine:
                    throw new Error("axis OXY is not serialized");
                case Line:
                    P1=(Integer)slots[4];
                    P2=(Integer)slots[5];
                    Model.addLine(primitiveId, Point.dbPoint.get(P1), Point.dbPoint.get(P2));
                    break;
                case Circle:
                    P1=(Integer)slots[4];
                    P2=(Integer)slots[5];
                    Model.addCircle(primitiveId, Point.dbPoint.get(P1), Point.dbPoint.get(P2));
                    break;
                case Arc:
                    P1=(Integer)slots[4];
                    P2=(Integer)slots[5];
                    //P3=(Integer)slots[6];
                    Model.addArc(primitiveId, Point.dbPoint.get(P1), Point.dbPoint.get(P2));
                    break;
            }
            lstate =STA_END;
            return;
        }

        throw new Error("invalid input line : " + input);
    }

    private void processConstraint(){
/*
 *      ID: 4;
 *      TYPE: Connect2Points;
 *      K: 2;
 *      L: 5;
 *      M: -1;
 *      N: -1;
 *      PARAM: -1;
 */
        if(PAT_ID.matcher(input).matches()) {
            slots[0] = Integer.parseInt(PAT_ID.matcher(input).group(1));
            return;
        }

        if(PAT_TYPE.matcher(input).matches()) {
            slots[3] = (PAT_TYPE.matcher(input).group(1));
            return;
        }


        if(PAT_K.matcher(input).matches()) {
            slots[7] = Integer.parseInt(PAT_K.matcher(input).group(1));
            return;
        }

        if(PAT_L.matcher(input).matches()) {
            slots[8] = Integer.parseInt(PAT_L.matcher(input).group(1));
            return;
        }

        if(PAT_M.matcher(input).matches()) {
            slots[9] = Integer.parseInt(PAT_M.matcher(input).group(1));
            return;
        }

        if(PAT_N.matcher(input).matches()) {
            slots[10] = Integer.parseInt(PAT_N.matcher(input).group(1));
            return;
        }

        if(PAT_PARAM.matcher(input).matches()) {
            slots[11] = Integer.parseInt(PAT_PARAM.matcher(input).group(1));
            return;
        }

        if(PAT_END.matcher(input).matches()) {
            /// CONSUME slots


            lstate=STA_END;
        }

        throw new Error("invalid input line : " + input);

    }

    /**
     * Skip all comments , and other markers but remember to check file format version.
     */
    private void processEnd(){

        if(input == null || input.trim().isEmpty()){
            // skip blank lines
            return;
        }
        if(PREFIX_COMMENTS.matcher(input).matches()){
            // skip comment input #!
            return;
        }
        if(VAR_FILE_FORMAT.matcher(input).matches()){
            String version=VAR_FILE_FORMAT.matcher(input).group(0);
            if(!version.equals(FILE_FORMAT_VERSION)) {
                throw new Error("unsupported format version ! =  "+version);
            }
            return;
        }
        if(MARKER_DEFINITION_BEGIN.matcher(input).matches()){
            // Begin Processing
            return;
        }

        if(MARKER_DEFINITION_END.matcher(input).matches()){
            // End Processing
            return;
        }
    }

}
