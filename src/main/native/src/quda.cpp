#include "quda.h"

#include <string>

namespace utility {

template <> void printer<int>(int i, int object) { printf("%d  %d \n", i, object); }

template <> void printer<double>(int i, double object) { printf("%d  %f\n", i, object); }


} // namespace utility