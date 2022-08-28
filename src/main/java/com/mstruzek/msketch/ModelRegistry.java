package com.mstruzek.msketch;

import java.util.Set;
import java.util.TreeMap;
import java.util.zip.CRC32C;

/**
 * Facade into Java, JNI implementation . All modified states method should forward registration and updates to JNI.
 */
public class ModelRegistry {

    public static TreeMap<Integer, Point> dbPoint = new TreeMap<Integer, Point>();
    /*** tablica wszystkich elemntow */
    public static TreeMap<Integer, GeometricObject> dbPrimitives = new TreeMap<Integer, GeometricObject>();
    /*** tablica wszystkich parametrow */
    public static TreeMap<Integer, Parameter> dbParameter = new TreeMap<Integer, Parameter>();
    /*** tablica wszystkich linii */
    public static TreeMap<Integer, Constraint> dbConstraint = new TreeMap<>();
    /*** next id for geometric primitive */
    public static int primitiveCounter = 0;
    /*** nex id for parameter */
    public static int parameterCounter = 0;
    /*** next id for constraint */
    public static int constraintCounter = 0;
    /*** next id for control point */
    public static int pointCounter = 0;

    public static int nextPointId() {
        return pointCounter++;
    }

    public static int nextPrimitiveId() {
        return primitiveCounter++;
    }

    public static int nextParameterId() {
        return parameterCounter++;
    }

    public static Integer nextConstraintId() {
        return constraintCounter++;
    }

    public static Integer nextConstraintId(Set<Integer> skipIdentifiers) {
        int nextId = constraintCounter++;
        while (skipIdentifiers.contains(nextId)) {
            nextId = constraintCounter++;
        }
        return nextId;
    }

    public static TreeMap<Integer, Point> dbPoint() {
        return dbPoint;
    }

    public static TreeMap<Integer, GeometricObject> dbPrimitives() {
        return dbPrimitives;
    }

    public static TreeMap<Integer, Parameter> dbParameter() {
        return dbParameter;
    }

    public static TreeMap<Integer, Constraint> dbConstraint() {
        return dbConstraint;
    }

    @JniModel
    public static void registerPoint(int id, Point point) {
        if (point.id >= 0 && !ModelRegistry.dbPoint.containsKey(point.id)) {
            dbPoint.put(id, point);
        }
    }

    @JniModel
    public static void registerConstraint(int constraintId, Constraint constraint) {
        dbConstraint.put(constraintId, constraint);
    }

    @JniModel
    public static GeometricObject registerGeometric(int primitiveId, GeometricObject geometricObject) {
        dbPrimitives.put(primitiveId, geometricObject);
        return geometricObject;
    }

    @JniModel
    public static void registerParameter(int parameterId, Parameter parameter) {
        dbParameter.put(parameterId, parameter);
    }

    @JniModel
    public static void setLocation(Integer pointId, double x, double y) {
        dbPoint.get(pointId).setLocation(x, y);

    }

    @JniModel
    public static void setParameterValue(int id, double parameterValue) {
        dbParameter.get(id).setValue(parameterValue);
        /// ---- update in JNI !
    }

    @JniModel
    public static void removeObjectsFromModel() {
        ModelRegistry.dbConstraint().clear();
        ModelRegistry.dbParameter().clear();
        ModelRegistry.dbPrimitives().clear();
        ModelRegistry.dbPoint().clear();

        ModelRegistry.constraintCounter = 0;
        ModelRegistry.parameterCounter = 0;
        ModelRegistry.primitiveCounter = 0;
        ModelRegistry.pointCounter = 0;

    }

    @JniModel
    public static void removeParameter(int parameterId) {
        dbParameter.remove(parameterId);
    }

    @JniModel
    public static void removeConstraint(int constraintId) {
        dbConstraint.remove(constraintId);
    }

    @JniModel
    public static void removePoint(int pointId) {
        dbPoint.remove(pointId);
    }

    @JniModel
    public static void removePrimitives(int primitiveId) {
        dbPrimitives.remove(primitiveId);
    }

    /**
     * All destructive/constructive changes applied into model.
     * - changes applied into model ( register/unregister )
     * @return checksum from model ids
     */
    public static long computationSnapshotId() {
        CRC32C checksum = new CRC32C();
        for (int pointId : ModelRegistry.dbPoint().keySet())
            checksum.update(pointId);
        for (int geometricId : ModelRegistry.dbPrimitives().keySet())
            checksum.update(geometricId * 100);
        for (int constraintId : ModelRegistry.dbConstraint().keySet())
            checksum.update(constraintId * 1000);
        for (int parameterId : ModelRegistry.dbParameter().keySet())
            checksum.update(parameterId * 10000);

        return checksum.getValue();
    }
}


