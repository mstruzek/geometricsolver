package com.mstruzek.msketch;

import java.lang.annotation.ElementType;
import java.lang.annotation.Target;

/**
 * Marker for methods that will synchronize data with JNI context.
 */
@Target(ElementType.METHOD)
public @interface JniModel {
}
