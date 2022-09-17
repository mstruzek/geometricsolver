#ifndef _UTILITY_CUH_
#define _UTILITY_CUH_

#include "cuda_runtime_api.h"

#include <memory>
#include <vector>
#include <xmemory>

#include "cuerror.h"

namespace utility {

template <typename Type> void mallocHost(Type **dest, size_t size) {
    checkCudaStatus(cudaMallocHost((void **)dest, size * sizeof(Type)));
    // * - ::cudaHostAllocMapped: Maps the allocation into the CUDA address space.
    // * - - The device pointer to the memory may be obtained by calling * ::cudaHostGetDevicePointer()
}

template <typename Type> void mallocAsync(Type **dest, size_t size, cudaStream_t stream) {
    checkCudaStatus(cudaMallocAsync((void **)dest, size * sizeof(Type), stream));
}

template <typename Type> void memcpyAsync(Type **dest_device, const Type *const &vector, size_t size, cudaStream_t stream) {
    /// transfer into new allocation host_vector
    checkCudaStatus(cudaMemcpyAsync(*dest_device, vector, size * sizeof(Type), cudaMemcpyHostToDevice, stream));
}

template <typename Type, template <typename> typename Allocator>
void memcpyAsync(Type **dest_device, std::vector<Type, Allocator<Type>> const &vector, cudaStream_t stream) {
    /// memcpy to device
    memcpyAsync(dest_device, vector.data(), vector.size(), stream);
}

template <typename Type, template <typename> typename Allocator> void memcpyFromDevice(std::vector<Type, Allocator<Type>> &vector, Type *src_device) {
    checkCudaStatus(cudaMemcpy(vector.data(), src_device, vector.size() * sizeof(Type), cudaMemcpyDeviceToHost));
}

template <typename Type, template <typename> typename Allocator>
void memcpyFromDevice(std::vector<Type, Allocator<Type>> &vector, Type *src_device, cudaStream_t stream) {
    /// transfer into new allocation host_vector
    checkCudaStatus(cudaMemcpyAsync(vector.data(), src_device, vector.size() * sizeof(Type), cudaMemcpyDeviceToHost, stream));
}

template <typename Type> void memcpyFromDevice(Type *dest, Type *src_device, size_t size, cudaStream_t stream) {
    /// transfer into new allocation
    checkCudaStatus(cudaMemcpyAsync(dest, src_device, size * sizeof(Type), cudaMemcpyDeviceToHost, stream));
}

template <typename Type> void memcpyOnDevice(Type *dest_dev, Type *src_dev, size_t size, cudaStream_t stream) {
    /// transfer into new allocation
    checkCudaStatus(cudaMemcpyAsync(dest_dev, src_dev, size * sizeof(Type), cudaMemcpyDeviceToDevice, stream));
}

template <typename Type> void freeAsync(Type *dev_ptr, cudaStream_t stream) {
    if (dev_ptr != nullptr)
        checkCudaStatus(cudaFreeAsync(dev_ptr, stream));
}

template <typename Type> void freeMemHost(Type **ptr) {
    if (*ptr != nullptr) {
        checkCudaStatus(cudaFreeHost(*ptr));
        *ptr = nullptr;
    }
}

template<typename Type> void memsetAsync(Type *dev_ptr, int value, size_t size, cudaStream_t stream) {
    ///
    checkCudaStatus(cudaMemsetAsync(dev_ptr, value, size * sizeof(Type), stream));
}


///  ------------------------------------------------------------------------------------------ ///
/// Operatory pomocnicze

#define INIT_VALUE 0

#define EMPTY_VALUE 0

template<typename Type> int uniqueElementId(const Type *element) { 
    return element->id; 
}

///  ------------------------------------------------------------------------------------------ ///

template<
    typename Accumulated, 
    typename Type, 
    typename ElementFunction = Accumulated (*)(Type const &)> 
struct map_reduce_element {

    map_reduce_element(Accumulated (*function)(Type const &)) : function(function) {}

    Accumulated operator()(Accumulated accumulated, Type const &element) { return accumulated + function(element); }

    ElementFunction function;
};


///  ------------------------------------------------------------------------------------------ ///

template<
    typename Input,     
    typename Output, 
    typename ObjIdFunction = int (*)(typename Input::value_type const *)>
void stateOffset(const Input &vector, ObjIdFunction objectIdFunction, Output &output) {
    if (vector.empty()) {
        return;
    }
    output = Output(objectIdFunction(&*vector.rbegin()) + 1, EMPTY_VALUE);
    auto iterator = vector.begin();
    int offset = 0;
    while (iterator != vector.end()) {
        auto objectId = objectIdFunction(&*iterator);
        output[objectId] = offset++;
        iterator++;
    }
}

///  ------------------------------------------------------------------------------------------ ///

/// 
/// Accumulated value, [ 0, E(0), E(0,1) + ... , E(0,N-1);  E(0,N)] ,  total accumulate value is E(0,N)
///
template<
    typename Vector, 
    typename ValueFunction, 
    typename Output>
void accumulatedValue(const Vector &vector, ValueFunction valueFunction, Output &accumulated) {
    int accValue = 0;
    accumulated = Output(vector.size() + 1, 0);
    for (int offset = 0; offset < vector.size(); offset++) {
        accumulated[offset] = accValue;
        accValue = accValue + valueFunction(vector[offset]);
    }
    accumulated[accumulated.size() - 1] = accValue;
}

///  ------------------------------------------------------------------------------------------ ///

/// 
///  fetche value of the last index if vector is not empty
/// 

template<
    typename Vector> 
int lastIndexValue(Vector const &vector) { 
    ///
    return vector.empty() ? 0 : vector[vector.size() - 1];
}


///  ------------------------------------------------------------------------------------------ ///

} // namespace utility

#endif // !_UTILITY_CUH_
