#pragma once

namespace settings
{

struct Settings
{   
    bool DEBUG_TENSOR_A             = false;        /// 1
    bool DEBUG_TENSOR_B             = false;        /// 2
    bool DEBUG                      = false;        /// 3    
    bool DEBUG_TENSOR_SV            = false;        /// 4    
    bool CLOCK_MILLISECONDS         = true;         /// 5
    bool CLOCK_NANOSECONDS          = true;         /// 6
    
    bool DEBUG_SOLVER_CONVERGENCE   = false;        /// 8
    bool DEBUG_CHECK_ARG            = false;        /// 9
                 
    int COMPUTATION_MODE            = 1;            /// 20  - 1 - DENSE_MODE , 2 - SPARSE_MODE , *3 - DIRECT_MODE
    int SOLVER_MODE                 = 1;            /// 21  - 1 - 
    bool SOLVER_INC_HESSIAN         = false;        /// 24

    double SOLVER_LWORK_FACTOR      = 1.0;          /// 26
    double SOLVER_EPSILON           = 10.0;         /// 27

    bool DEBUG_CSR_FORMAT           = false;        /// 30
    bool DEBUG_COO_FORMAT           = false;        /// 31


    bool STREAM_CAPTURING = false; /// 60
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
