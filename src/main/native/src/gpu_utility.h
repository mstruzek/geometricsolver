#ifndef _GPU_UTILITY_H_
#define _GPU_UTILITY_H_

#include <string>

#include "cusolverSp.h"

#ifndef max
#define max(a, b) ((a) > (b)) ? (a) : (b)
#endif

#ifndef min
#define min(a, b) ((a) > (b)) ? (b) : (a)
#endif



struct CusolverStatusName {
  
    const char *CU_SOLVER_STATUS_NAME[32] = {};
    
    CusolverStatusName();
};

/// <summary>
/// utility function simmillar to those error handling functions in cublas, cusparse
/// </summary>
/// <param name="status"></param>
/// <returns></returns>
const char *cusolverGetErrorName(cusolverStatus_t status);

void log_error(cusparseStatus_t status, const char *error);


#endif // _GPU_UTILITY_H_