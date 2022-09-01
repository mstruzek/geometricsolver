#include "settings.h"

#include <stdio.h>

namespace settings
{

/// singletone object
static Settings data = {};

Settings *get()
{
    return &data;
}

int setBooleanProperty(int id, bool value)
{
    Settings *setting = &data;
    switch (id)
    {
    case 0:
        setting->DEBUG = value;             // 0
        break;
    case 1:
        setting->KERNEL_PRE = value;      // 1     
        break;
    case 2:
        setting->DEBUG_TENSOR_A = value;    // 2
        break;
    case 3:
        setting->DEBUG_TENSOR_B = value;    // 3
        break;
    case 4:
        setting->DEBUG_TENSOR_SV = value;   // 4
        break;
    case 6:
        setting->CLOCK_NANOSECONDS = value; // 6
        break;
    case 7:
        setting->SOLVER_INC_HESSIAN = value; // 7
        break;
    case 8:
        setting->DEBUG_SOLVER_CONVERGENCE = value; // 8
        break;
    case 9:
        setting->DEBUG_CHECK_ARG = value;           // 9
        break;
    default:
        printf("[error] bool property not found , id ( %d ) value( %d ) !!\n", id, value);
        return 1;
    }
    return 0;
}

/**
 * Set long settings property.
 *
 * GRID_SIZE
 * BLOCK_SIZE
 *
 * @param id
 * @param value
 * @return
 */

int setLongProperty(int id, long value)
{
    Settings *config = &data;
    switch (id)
    {
    case 10:
        config->GRID_SIZE = value;  // 10
        break;
    case 11:
        config->BLOCK_SIZE = value; // 11
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

int setDoubleProperty(int id, double value)
{
    Settings *config = &data;
    switch (id)
    {
    case 21:
        config->CU_SOLVER_LWORK_FACTOR = value;
        break;
    case 22:
        config->CU_SOLVER_EPSILON = value;
        break;
    default:
        printf("[error] double property not found , id ( %d ) value( %e ) !!\n", id, value);
        return 1;
    }
    return 0;
}

} // namespace settings