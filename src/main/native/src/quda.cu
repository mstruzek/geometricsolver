#include "quda.cuh"

#define KERNEL_EXECUTOR

namespace quda {

template <> __device__ __host__ const char *line_format<int>() {
    static const char *int_format = "(%d)  %d, \n";
    return int_format;
}

template <> __device__ __host__ const char *line_format<double>() {
    static const char *doubleFormat = "(%d)  %f, \n";
    return doubleFormat;
}


template <typename Type> __global__ void __stdout_vector_kernel__(Type *vector, int size) {
    const char *format = line_format<Type>();
    int i = size;
    while (i-- > 0) {
        log(format, i, vector[i]);
    }
}

/// <summary>
/// Device kernel executor for debug stdout from device.
/// </summary>
/// <typeparam name="Type"></typeparam>
/// <param name="vector"></param>
/// <param name="size"></param>

KERNEL_EXECUTOR template<typename Type> void stdout_vector_kernel(Type *vector, int size) {
    //
    const unsigned GRID_DIM = 1;
    const unsigned BLOCK_DIM = 1;
    const unsigned STREAM = 0;
    const unsigned NS = 0;

    __stdout_vector_kernel__<Type><<<GRID_DIM, BLOCK_DIM, NS, STREAM>>>(vector, size);
}

template void stdout_vector_kernel<double>(double *vector, int size);
template void stdout_vector_kernel<int>(int *vector, int size);

} // namespace quda