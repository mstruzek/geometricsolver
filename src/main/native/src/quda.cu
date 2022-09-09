#include "quda.cuh"

namespace quda {

template <> __device__ __host__ const char *line_format<int>() {
    static const char *int_format = "(%d)  %d, \n";
    return int_format;
}

template <> __device__ __host__ const char *line_format<double>() {
    static const char *doubleFormat = "(%d)  %f, \n";
    return doubleFormat;
}


template <typename Type> __global__ void stdout_vector_kernel(Type *vector, int size) {
    const char *format = line_format<Type>();
    int i = size;
    while (i-- > 0) {
        log(format, i, vector[i]);
    }
}

} // namespace quda