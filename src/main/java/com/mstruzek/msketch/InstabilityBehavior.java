package com.mstruzek.msketch;

import java.lang.annotation.ElementType;
import java.lang.annotation.Target;

/**
 * Mathematical model behavior instability in solution space.
 */
@Target({ElementType.METHOD, ElementType.TYPE})
public @interface InstabilityBehavior {

    String description()  default "";
}
