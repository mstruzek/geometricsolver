#include "settings.h"

#include <stdio.h>

namespace settings {

/// singletone object
static Settings data = {};

Settings *get() { return &data; }

int setBooleanProperty(int id, bool value) {
    Settings *setting = &data;
    switch (id) {
    case 1:
        setting->DEBUG_TENSOR_A = value;    //
        break;
    case 2:
        setting->DEBUG_TENSOR_B = value;    //
        break;
    case 3:
        setting->DEBUG = value;             //
        break;
    case 4:
        setting->DEBUG_TENSOR_SV = value;   // 
        break;
    case 5:
        setting->CLOCK_MILLISECONDS= value; //
        break;
    case 6:
        setting->CLOCK_NANOSECONDS = value; // 
        break;
    case 8:
        setting->DEBUG_SOLVER_CONVERGENCE = value; // 
        break;
    case 9:
        setting->DEBUG_CHECK_ARG = value;   //
        break;
    case 24:
        setting->SOLVER_INC_HESSIAN = value; // 
        break;
    case 30:
        setting->DEBUG_CSR_FORMAT = value; //
        break;
    case 31:
        setting->DEBUG_COO_FORMAT= value; // 
        break;
    case 60:
        setting->STREAM_CAPTURING = value; // 
        break;
    default:
        printf("[error] bool property not found , id ( %d ) value( %d ) !!\n", id, value);
        return 1;
    }
    return 0;
}

/**
 * Set settings from long property.
 *
 * @param id
 * @param value
 * @return
 */

int setLongProperty(int id, long value) {
    Settings *config = &data;
    switch (id) {
    case 20:
        config->COMPUTATION_MODE = value; // 
        break;
    case 21:
        config->SOLVER_MODE= value; // 
        break;

    default:
        printf("[error] long property not found , id ( %d ) value( %d ) !!\n", id, value);
        return 1;
    }
    return 0;
}

/**
 * Set double settings property.
 *
 * CU_SOLVER_LWORK_FACTOR
 *
 * @param id
 * @param value
 * @return
 */

int setDoubleProperty(int id, double value) {
    Settings *config = &data;
    switch (id) {
    case 26:
        config->SOLVER_LWORK_FACTOR = value;
        break;
    case 27:
        config->SOLVER_EPSILON = value;
        break;
    default:
        printf("[error] double property not found , id ( %d ) value( %e ) !!\n", id, value);
        return 1;
    }
    return 0;
}

} // namespace settings