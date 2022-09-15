#ifndef _UTILITY_CUH_
#define _UTILITY_CUH_

#include "cuda_runtime_api.h"

#include <memory>
#include <vector>

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

template <typename Type>
void memcpyAsync(Type **dest_device, const Type *const &vector, size_t size, cudaStream_t stream) {
    /// transfer into new allocation host_vector
    checkCudaStatus(cudaMemcpyAsync(*dest_device, vector, size * sizeof(Type), cudaMemcpyHostToDevice, stream));
}

template <typename Type, template <typename> typename Allocator>
void memcpyAsync(Type **dest_device, std::vector<Type, Allocator<Type>> const &vector, cudaStream_t stream) {
    /// memcpy to device
    memcpyAsync(dest_device, vector.data(), vector.size(), stream);
}


template <typename Type, template <typename> typename Allocator>
void memcpyFromDevice(std::vector<Type, Allocator<Type>> &vector, Type *src_device) {
    checkCudaStatus(cudaMemcpy(vector.data(), src_device, vector.size() * sizeof(Type), cudaMemcpyDeviceToHost));
}

template <typename Type, template <typename> typename Allocator>
void memcpyFromDevice(std::vector<Type, Allocator<Type>> &vector, Type *src_device, cudaStream_t stream) {
    /// transfer into new allocation host_vector
    checkCudaStatus(
        cudaMemcpyAsync(vector.data(), src_device, vector.size() * sizeof(Type), cudaMemcpyDeviceToHost, stream));
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

template <typename Type> void memsetAsync(Type *dev_ptr, int value, size_t size, cudaStream_t stream) {
    ///
    checkCudaStatus(cudaMemsetAsync(dev_ptr, value, size * sizeof(Type), stream));
}

template <typename Type, template <typename> typename Allocator, typename ObjIdFunction>
std::vector<int, Allocator<int>> stateOffset(const std::vector<Type, Allocator<Type>> &vector,
                                             ObjIdFunction objectIdFunction) {
    if (vector.empty()) {
        return std::vector<int, Allocator<int>>(0);
    }

    std::vector<int, Allocator<int>> offsets(objectIdFunction(vector.rbegin()) + 1, 0);
    auto iterator = vector.begin();
    int offset = 0;
    while (iterator != vector.end()) {
        auto objectId = objectIdFunction(iterator);
        offsets[objectId] = offset++;
        iterator++;
    }
    return offsets;
}

/// #/include <numeric> std::partial_sum

/// Accumulated value, [ 0, E(0), E(0,1) + ... , E(0,N-1);  E(0,N)] ,  total accumulate value is E(0,N)
///
template <typename Type, template <typename> typename Allocator, typename ValueFunction>
std::vector<int, Allocator<int>> accumulatedValue(const std::vector<Type, Allocator<Type>> &vector,
                                                  ValueFunction valueFunction) {
    int accValue = 0;
    std::vector<int, Allocator<int>> accumulated(vector.size() + 1, 0);
    for (int offset = 0; offset < vector.size(); offset++) {
        accumulated[offset] = accValue;
        accValue = accValue + valueFunction(vector[offset]);
    }
    accumulated[accumulated.size() - 1] = accValue;
    return accumulated;
}

} // namespace utility

#endif // !_UTILITY_CUH_
