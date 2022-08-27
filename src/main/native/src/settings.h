#pragma once

namespace settings
{

struct Settings
{
    bool DEBUG                      = false;        // 0
    bool DEBUG_TENSOR_A             = false;        // 2
    bool DEBUG_TENSOR_B             = false;        // 3
    bool DEBUG_TENSOR_SV            = false;        // 4
    bool CLOCK_MILLISECONDS         = true;         // 5
    bool CLOCK_NANOSECONDS          = false;        // 6
    bool SOLVER_INC_HESSIAN         = false;        // 7
    bool DEBUG_SOLVER_CONVERGENCE   = false;        // 8
    bool DEBUG_CHECK_ARG            = false;        // 9

    size_t GRID_SIZE        = 1;    // 10
    size_t BLOCK_SIZE       = 512;  // 11

    double CU_SOLVER_LWORK_FACTOR = 1.0;            // 21
};

/**
 * Get default settings setting.
 *
 */
Settings const * get();

/**
 * Set boolean settings property.
 * DEBUG
 * DEBUG_MATRIX_A
 * DEBUG_TENSOR_B
 * DEBUG_TENSOR_SV
 * CLOCK_MILLISECONDS
 * CLOCK_NANOSECONDS
 * SOLVER_INC_HESSIAN
 * DEBUG_SOLVER_CONVERGENCE
 *
 * @param id
 * @param value
 * @return
 */

int setBooleanProperty(int id, bool value);

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

int setLongProperty(int id, long value);

/**
 * Set double settings property.
 *
 * CU_SOLVER_LWORK_FACTOR
 *
 * @param id
 * @param value
 * @return
 */

int setDoubleProperty(int id, double value);

} // namespace settings
