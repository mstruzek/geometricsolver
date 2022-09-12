#include "quda.cuh"

#include <string>

#define KERNEL_EXECUTOR

namespace quda {




template <> struct printer<int> {
    __device__ __host__ void operator()(int i, const int &object) { 
        ///
        printf("%d  %d \n", i, object);
    }
};

template <> struct printer<double> {
    __device__ __host__ void operator()(int i, const double &object) {
        printf("%d  %f\n", i , object);
    }
};



template <typename Type> __global__ void __stdout_vector_kernel__(Type *vector, int size) {
    printer<Type> printer;
    int i = size;
    while (i-- > 0) {
        printer(i, vector[i]);
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