#ifndef _UTILITY_CUH_
#define _UTILITY_CUH_

#include "cuda_runtime_api.h"

#include "cuerror.h"

#include <vector>
#include <memory>


namespace utility
{

template <typename Ty> void mallocHost(Ty **dest, size_t size)
{
    checkCudaStatus(cudaMallocHost((void **)dest, size * sizeof(Ty), cudaHostAllocDefault));
    // * - ::cudaHostAllocMapped: Maps the allocation into the CUDA address space.
    // * - - The device pointer to the memory may be obtained by calling * ::cudaHostGetDevicePointer()
}

template <typename Ty> void mallocAsync(Ty **dest, size_t size)
{
    checkCudaStatus(cudaMallocAsync((void **)dest, size * sizeof(Ty), stream));
}

template <typename Ty> void memcpyToDevice(Ty **dest_device, const Ty *const &vector, size_t size)
{
    /// transfer into new allocation host_vector
    checkCudaStatus(cudaMemcpyAsync(*dest_device, vector, size * sizeof(Ty), cudaMemcpyHostToDevice, stream));
}

template <typename Ty> void memcpyToDevice(Ty **dest_device, std::vector<Ty> const &vector)
{
    /// memcpy to device
    memcpyToDevice(dest_device, vector.data(), vector.size());
}

template <typename Ty> void mallocToDevice(Ty **dev_ptr, size_t size)
{
    /// safe malloc
    checkCudaStatus(cudaMallocAsync((void **)dev_ptr, size, stream));
}

template <typename Ty> void memcpyFromDevice(std::vector<Ty> &vector, Ty *src_device)
{
    /// transfer into new allocation host_vector
    checkCudaStatus(
        cudaMemcpyAsync(vector.data(), src_device, vector.size() * sizeof(Ty), cudaMemcpyDeviceToHost, stream));
}

template <typename Ty> void memcpyFromDevice(Ty *dest, Ty *src_device, size_t arity)
{
    /// transfer into new allocation
    checkCudaStatus(cudaMemcpyAsync(dest, src_device, arity * sizeof(Ty), cudaMemcpyDeviceToHost, stream));
}

template <typename Ty> void freeMem(Ty *dev_ptr)
{
    /// safe free mem
    checkCudaStatus(cudaFreeAsync(dev_ptr, stream));
}

template <typename Ty> void freeHostMem(Ty *ptr)
{
    /// safe free mem
    checkCudaStatus(cudaFreeHost(ptr));
}

template <typename Ty> void memset(Ty *dev_ptr, int value, size_t size)
{
    ///
    checkCudaStatus(cudaMemsetAsync(dev_ptr, value, size * sizeof(Ty), stream));
}

template <typename Obj, typename ObjIdFunction>
std::unique_ptr<int[]> stateOffset(std::vector<Obj> objects, ObjIdFunction objectIdFunction)
{
    if (objects.empty())
    {
        return std::unique_ptr<int[]>(new int[0]);
    }

    std::unique_ptr<int[]> offsets(new int[objectIdFunction(++objects.rbegin())]);
    auto iterator = objects.begin();
    int offset = 0;
    while (iterator != objects.end())
    {
        auto objectId = objectIdFunction(iterator);
        offsets[objectId] = offset++;
        iterator++;
    }
    return offsets;
}

/// #/include <numeric> std::partial_sum

template <typename Obj, typename ValueFunction>
std::unique_ptr<int[]> accumalatedValue(std::vector<Obj> vector, ValueFunction valueFunction)
{
    int accValue = 0;
    std::unique_ptr<int[]> accumulated(new int[vector.size()]);
    for (int offset = 0; offset < vector.size(); offset++)
    {
        accumulated[offset] = accValue;
        accValue = accValue + valueFunction(vector[offset]);
    }
    return accumulated;
}

} // namespace utility


#endif // !_UTILITY_CUH_
