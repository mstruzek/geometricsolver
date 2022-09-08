#pragma once

namespace settings
{

struct Settings
{
    bool DEBUG                      = false;        /// 0
    bool KERNEL_PRE                 = true;         /// 1
    bool DEBUG_TENSOR_A             = false;        /// 2
    bool DEBUG_TENSOR_B             = false;        /// 3
    bool DEBUG_TENSOR_SV            = false;        /// 4
    bool CLOCK_NANOSECONDS          = true;         /// 6
    bool SOLVER_INC_HESSIAN         = false;        /// 7
    bool DEBUG_SOLVER_CONVERGENCE   = false;        /// 8
    bool DEBUG_CHECK_ARG            = false;        /// 9
    bool STREAM_CAPTURING           = false;        /// 100

    unsigned GRID_SIZE              = 1;            /// 10
    unsigned BLOCK_SIZE             = 512;          /// 11
    unsigned COMPUTATION_MODE       = 1;            /// 12  - 1 - DENSE_LAYOUT , 2 - SPARSE_LAYOUT , 3 - DIRECT_LAYOUT

    double CU_SOLVER_LWORK_FACTOR   = 1.0;          /// 21
    double CU_SOLVER_EPSILON        = 10.0;         /// 22
};

/**
 * Get default settings setting.
 *
 */
Settings *get();

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
