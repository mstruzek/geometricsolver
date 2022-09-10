#ifndef _GPU_UTILITY_H_
#define _GPU_UTILITY_H_

#include <string>

#include "cusolverSp.h"



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

void fail(cusparseStatus_t status, const char *error);


#endif // _GPU_UTILITY_H_