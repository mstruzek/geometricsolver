#ifndef _UTILITY_CUH_
#define _UTILITY_CUH_

#include "cuda_runtime_api.h"


#include <memory>
#include <vector>

#include "cuerror.h"


namespace utility {

template <typename Ty> void mallocHost(Ty **dest, size_t size) {
    checkCudaStatus(cudaMallocHost((void **)dest, size * sizeof(Ty), cudaHostAllocDefault));
    // * - ::cudaHostAllocMapped: Maps the allocation into the CUDA address space.
    // * - - The device pointer to the memory may be obtained by calling * ::cudaHostGetDevicePointer()
}

template <typename Ty> void mallocAsync(Ty **dest, size_t size, cudaStream_t stream) {
    checkCudaStatus(cudaMallocAsync((void **)dest, size * sizeof(Ty), stream));
}

template <typename Ty>
void memcpyAsync(Ty **dest_device, const Ty *const &vector, size_t size, cudaStream_t stream) {
    /// transfer into new allocation host_vector
    checkCudaStatus(cudaMemcpyAsync(*dest_device, vector, size * sizeof(Ty), cudaMemcpyHostToDevice, stream));
}

template <typename Ty> void memcpyAsync(Ty **dest_device, std::vector<Ty> const &vector, cudaStream_t stream) {
    /// memcpy to device
    memcpyAsync(dest_device, vector.data(), vector.size(), stream);
}

template <typename Ty> void memcpyFromDevice(std::vector<Ty> &vector, Ty *src_device, cudaStream_t stream) {
    /// transfer into new allocation host_vector
    checkCudaStatus(
        cudaMemcpyAsync(vector.data(), src_device, vector.size() * sizeof(Ty), cudaMemcpyDeviceToHost, stream));
}

template <typename Ty> void memcpyFromDevice(Ty *dest, Ty *src_device, size_t size, cudaStream_t stream) {
    /// transfer into new allocation
    checkCudaStatus(cudaMemcpyAsync(dest, src_device, size * sizeof(Ty), cudaMemcpyDeviceToHost, stream));
}


template <typename Ty> void freeAsync(Ty *dev_ptr, cudaStream_t stream) {
    if (dev_ptr != nullptr) checkCudaStatus(cudaFreeAsync(dev_ptr, stream));    
}

template <typename Ty> void freeMemHost(Ty **ptr) {
    if (*ptr != nullptr) {
        checkCudaStatus(cudaFreeHost(*ptr));
        *ptr = nullptr;
    }
}

template <typename Ty> void memsetAsync(Ty *dev_ptr, int value, size_t size, cudaStream_t stream) {
    ///
    checkCudaStatus(cudaMemsetAsync(dev_ptr, value, size * sizeof(Ty), stream));
}

template <typename Obj, typename ObjIdFunction>
std::vector<int> stateOffset(std::vector<Obj> objects, ObjIdFunction objectIdFunction) {
    if (objects.empty()) {
        return std::vector<int>(0);
    }

    std::vector<int> offsets(objectIdFunction(objects.rbegin()) + 1, 0);
    auto iterator = objects.begin();
    int offset = 0;
    while (iterator != objects.end()) {
        auto objectId = objectIdFunction(iterator);
        offsets[objectId] = offset++;
        iterator++;
    }
    return offsets;
}

/// #/include <numeric> std::partial_sum

template <typename Obj, typename ValueFunction>
std::vector<int> accumalatedValue(std::vector<Obj> vector, ValueFunction valueFunction) {
    int accValue = 0;
    std::vector<int> accumulated(vector.size(), 0);
    for (int offset = 0; offset < vector.size(); offset++) {
        accumulated[offset] = accValue;
        accValue = accValue + valueFunction(vector[offset]);
    }
    return accumulated;
}

} // namespace utility

#endif // !_UTILITY_CUH_
