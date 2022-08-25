package com.mstruzek.jni;

public class OsUtility {

    static final String CONST_OS_WIN = "win";

    public static boolean isWindows() {
        return System.getProperty("os.name").toLowerCase().equals(CONST_OS_WIN);
    }

}
