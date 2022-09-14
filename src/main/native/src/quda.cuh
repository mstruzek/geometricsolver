#ifndef _QUDA_CUH_
#define _QUDA_CUH_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>

#include "cuerror.h"

#include "gpu_allocator.h"

#include "quda.cuh"

namespace utility {

template <typename Type> class dev_vector;

/// ===================================================== ===================================================== ///

/// <summary>
/// std::vector with characteristics to supply additional memmory from cuda Pinned Memory Allocator.
/// </summary>
template <typename Type> using cu_vector = std::vector<Type, gpu_allocator<Type>>;

/// ===================================================== ===================================================== ///

template <typename Type> class host_vector {
  public:
    /// <summary>
    /// Allocate owned pinned memory extent for requested type and size
    /// </summary>
    /// <param name="size"></param>
    host_vector(size_t size) : size(size), owner(true) {
        cudaError_t status = cudaMallocHost((void **)&memory, size * sizeof(Type));
        if (status != cudaSuccess) {
            fprintf(stderr, "[host_vector] vector memory allocation failed !\n");
            exit(1);
        }
    }

    /// <summary>
    /// Non owned reference view to host memory.
    /// </summary>
    /// <param name="source"></param>
    host_vector(const std::vector<Type> &source) : host_vector(source.data(), source.size()) {}

    /// <summary>
    /// Non owned reference view to host memory.
    /// </summary>
    /// <param name="memory"></param>
    /// <param name="size"></param>
    host_vector(Type *memory, size_t size) : owner(false), memory(memory), size(size) {}

    /// <summary>
    /// Copy host vector possible initialized with cuda pinned memory.
    /// </summary>
    /// <param name="src"></param>
    /// <param name="stream"></param>
    /// <returns></returns>
    host_vector<Type> &memcpy_of(host_vector<Type> &src, cudaStream_t stream = nullptr) {
        if (size != src.get_size()) {
            fprintf(stderr, "[dev_vector] allocation buffers on device and host are not identical !\n");
            exit(1);
        }
        cudaError_t status;
        if (stream) {
            status = cudaMemcpyAsync(memory, src.data(), sizeof(Type) * size, cudaMemcpyHostToHost, stream);
        } else {
            status = cudaMemcpy(memory, src.data(), sizeof(Type) * size, cudaMemcpyHostToHost);
        }
        if (status != cudaSuccess) {
            const char *error_name = cudaGetErrorName(status);
            const char *error_str = cudaGetErrorString(status);
            fprintf(stderr, " [host_vector] memcpy from host to host failed !( %s )  %s \n", error_name, error_str);
            exit(1);
        }
        return *this;
    }

    /// <summary>
    /// Copy device vector on the selected cuda stream asynchronously.
    /// </summary>
    /// <param name="src"></param>
    /// <param name="stream"></param>
    /// <returns></returns>
    host_vector<Type> &memcpy_of(dev_vector<Type> &src, cudaStream_t stream = nullptr) {
        if (size != src.get_size()) {
            fprintf(stderr, "[dev_vector] allocation buffers on device and host are not identical !\n");
            exit(1);
        }
        cudaError_t status;
        if (stream) {
            status = cudaMemcpyAsync(memory, src.data(), sizeof(Type) * size, cudaMemcpyDeviceToHost, stream);
        } else {
            status = cudaMemcpy(memory, src.data(), sizeof(Type) * size, cudaMemcpyDeviceToHost);
        }
        if (status != cudaSuccess) {
            const char *error_name = cudaGetErrorName(status);
            const char *error_str = cudaGetErrorString(status);
            fprintf(stderr, " [host_vector] memcpy from device failed !( %s )  %s \n", error_name, error_str);
            exit(1);
        }
        return *this;
    }

    /// <summary>
    ///  Type safe implicit conversion function.
    /// </summary>
    /// <returns></returns>
    operator Type *() { return data(); }

    /// <summary>
    /// Non-type safe implicit conversion function. opaque.
    /// </summary>
    /// <returns></returns>
    operator void *() { return data(); }

    /// <summary>
    /// Type safe memory reference handler.
    /// </summary>
    /// <returns></returns>
    Type *data() { return memory; }

    /// <summary>
    /// Subscripted reference into element.
    /// </summary>
    /// <typeparam name="Type"></typeparam>
    Type &operator[](const size_t index) noexcept { return memory[index]; }

    /// <summary>
    /// Allocation size in units of basic type - allocation/sizeof(Type)
    /// </summary>
    /// <returns></returns>
    size_t get_size() const { return size; }

    /// release memory into cuda driver only when in possession of this memory extent
    ~host_vector() {
        if (!owner)
            return;
        if (memory) {
            cudaError_t status = cudaFreeHost(memory);
            if (status != cudaSuccess) {
                const char *errorName = cudaGetErrorName(status);
                const char *errorStr = cudaGetErrorString(status);
                fprintf(stderr, "[host_vector] memory deallocation failed ; ( %s ) %s !\n", errorName, errorStr);
            }
        }
    }

  private:
    
    /// in charge of this memory extent or view only borrowed ( false )
    bool owner;

    /// memory extent in possession or view only
    Type *memory;

    /// allocation size in units sizeof(Type)
    size_t size;

};

/// ===================================================== ===================================================== ///
template <typename Type> class dev_vector {
  public:
    /// <summary>
    /// No resources attached.
    /// </summary>
    dev_vector() : memory(NULL), size(0), stream(NULL), owner(false) {}

    /// <summary>
    /// Take a snapshot of reference device vector.
    /// </summary>
    /// <param name="other"></param>
    dev_vector(dev_vector const &other) { *this = other; };

    /// <summary>
    /// Become a new owner of other memory allocation.
    /// </summary>
    /// <param name="other"></param>
    dev_vector(dev_vector &&other) : dev_vector() { *this = std::move(other); };

    /// <summary>
    /// setup non ownig reference to device memory extent
    /// </summary>
    /// <param name="memory"></param>
    /// <param name="size"></param>
    /// <param name="stream"></param>
    dev_vector(Type *memory, size_t size, cudaStream_t stream) : memory(memory), size(size), stream(stream), owner(false) {}

    /// <summary>
    /// Helper utility for pinned memory vector.
    /// </summary>
    /// <param name="vector"></param>
    /// <param name="stream"></param>
    dev_vector(cu_vector<Type> &vector, cudaStream_t stream) : dev_vector(vector.size(), stream) {
        memcpy_of(host_vector<Type>(vector.data(), vector.size()), stream);
    }

    /// <summary>
    /// Initialize owning memory extent possibly allocated on selected stream.
    /// </summary>
    /// <param name="size"></param>
    /// <param name="stream"></param>
    dev_vector(size_t size, cudaStream_t stream = nullptr) : size(size), stream(stream), owner(true) {
        cudaError_t status;
        if (stream) {
            status = cudaMallocAsync((void **)&memory, size * sizeof(Type), stream);
        } else {
            status = cudaMalloc((void **)&memory, size * sizeof(Type));
        }
        if (status != cudaSuccess) {
            const char *errorName = cudaGetErrorName(status);
            const char *errorStr = cudaGetErrorString(status);
            fprintf(stderr, "[dev_vector] vector memory allocation failed ; ( %s )  %s \n", errorName, errorStr);
            exit(1);
        }
    }

    dev_vector(int size, cudaStream_t stream = nullptr) : dev_vector(static_cast<size_t>(size), stream) {}

    /// <summary>
    /// Copy host vector on selected cuda stream asynchronously into device.
    /// </summary>
    /// <param name="src"></param>
    /// <param name="stream"></param>
    /// <returns></returns>
    dev_vector<Type> &memcpy_of(host_vector<Type> &src, cudaStream_t stream = nullptr) {
        if (size != src.get_size()) {
            fprintf(stderr, "[dev_vector] equivalent device buffer allocation in size required on device from host transfer !\n");
            exit(-1);
        }
        cudaError_t status;
        if (stream) {
            status = cudaMemcpyAsync(memory, src.data(), sizeof(Type) * src.get_size(), cudaMemcpyHostToDevice, stream);
        } else {
            status = cudaMemcpy(memory, src.data(), sizeof(Type) * src.get_size(), cudaMemcpyHostToDevice);
        }
        if (status != cudaSuccess) {
            const char *errorName = cudaGetErrorName(status);
            const char *errorStr = cudaGetErrorString(status);
            fprintf(stderr, "[dev_vector] memcpy into device failed ; ( %s )  %s \n", errorName, errorStr);
            exit(-1);
        }
        return *this;
    }

    /// <summary>
    /// Copy device vector on selected cuda stream asynchronously into device.
    /// </summary>
    /// <param name="src"></param>
    /// <param name="stream"></param>
    /// <returns></returns>
    dev_vector<Type> &memcpy_of(dev_vector<Type> &src, cudaStream_t stream = nullptr) {
        if (size != src.get_size()) {
            fprintf(stderr, "[dev_vector] equivalent device buffer allocation in size required on device ! \n");
            exit(-1);
        }
        cudaError_t status;
        if (stream) {
            status = cudaMemcpyAsync(memory, src.data(), sizeof(Type) * size, cudaMemcpyDeviceToDevice, stream);
        } else {
            status = cudaMemcpy(memory, src.data(), sizeof(Type) * size, cudaMemcpyDeviceToDevice);
        }
        if (status != cudaSuccess) {
            const char *errorName = cudaGetErrorName(status);
            const char *errorStr = cudaGetErrorString(status);
            fprintf(stderr, "[dev_vector] memcpy on device failed ; ( %s )  %s \n", errorName, errorStr);
            exit(-1);
        }
        return *this;
    }

    /// <summary>
    /// Copy from pinned memory allocation.
    /// </summary>
    /// <param name="vector"></param>
    /// <param name="stream"></param>
    /// <returns></returns>
    dev_vector<Type> &memcpy_of(cu_vector<Type> const &vector, cudaStream_t stream = nullptr) {
        if (size != vector.size()) {
            fprintf(stderr, "[dev_vector] equivalent device buffer allocation in size required on device from host transfer! \n");
            exit(-1);
        }
        cudaError_t status;
        if (stream) {
            status = cudaMemcpyAsync(memory, vector.data(), sizeof(Type) * vector.size(), cudaMemcpyHostToDevice, stream);
        } else {
            status = cudaMemcpy(memory, vector.data(), sizeof(Type) * vector.size(), cudaMemcpyHostToDevice);
        }
        if (status != cudaSuccess) {
            const char *errorName = cudaGetErrorName(status);
            const char *errorStr = cudaGetErrorString(status);
            fprintf(stderr, "[dev_vector] memcpy into device failed ; ( %s )  %s \n", errorName, errorStr);
            exit(-1);
        }
        return *this;
    }

    /// <summary>
    /// "One-way". Safely borrow extent or become owner of device vector.
    /// </summary>
    /// <param name="other"></param>
    /// <returns></returns>
    dev_vector &operator=(dev_vector &&other) {
        if (owner) {
            this->~dev_vector();
        }
        owner = std::exchange(other.owner, false);
        memory = std::exchange(other.memory, nullptr);
        size = std::exchange(other.size, 0);
        stream = std::exchange(other.stream, nullptr);
        return *this;
    }

    /// <summary>
    /// Take snapshot of reference object. No owner changes.
    /// </summary>
    /// <param name="other"></param>
    /// <returns></returns>
    dev_vector &operator=(dev_vector const &other) {
        if (owner) {
            this->~dev_vector();
        }
        owner = false;
        memory = other.memory;
        size = other.size;
        stream = other.stream;
        return *this;
    }

    /// <summary>
    /// Type safe implicit conversion function.
    /// </summary>
    /// <returns></returns>
    operator Type *() { return data(); }

    /// <summary>
    /// Non-type safe memory reference handler.
    /// </summary>
    /// <returns></returns>
    operator void *() { return data(); }

    /// <summary>
    /// Type safe memory reference handler.
    /// </summary>
    /// <returns></returns>
    Type *data() { return memory; }

    /// <summary>
    /// Allocation size in units of basic type - allocation/sizeof(Type)
    /// </summary>
    /// <returns></returns>
    size_t get_size() const { return size; }

    /// <summary>
    /// Associated stream of allocation for this extent, also used as default in dtor.
    /// </summary>
    /// <returns></returns>
    cudaStream_t get_stream() { return stream; }

    /// owned memory release into stream of origin 
    ~dev_vector() {
        if (!owner) {
            return;
        }        
        if (memory) {
            cudaError_t status;
            if (stream) {
                status = cudaFreeAsync(memory, stream);
            } else {
                status = cudaFree(memory);
            }
            if (status != cudaSuccess) {
                const char *errorName = cudaGetErrorName(status);
                const char *errorStr = cudaGetErrorString(status);
                fprintf(stderr, "[dev_vector] memory deallocation failed ! ( %s ) %s \n", errorName, errorStr);
            }
        }
    }

  private:
    /// in charge of this memory extent or view only borrowed ( false )
    bool owner;

    /// memory extent in possession or view only
    Type *memory;

    /// allocation size in units sizeof(Type)
    size_t size;

    /// stream of orgin for this memory extent  - MallocAsync/FreeAsync
    cudaStream_t stream;
};

/// ===================================================== ===================================================== ///

template <typename... Type> __device__ __host__ void log(const char *format, Type const &...values) { ::printf(format, values...); }

/// ===================================================== ===================================================== ///

template <typename Type> struct printer {};

template <typename Type> void stdout_vector_kernel(cudaStream_t stream, Type *vector, int size);

template <typename Type> void stdout_vector(dev_vector<Type> &vector, const char *title) {
    log("%s --- --- ---\n", title);
    stdout_vector_kernel<Type>(vector.get_stream(), vector.data(), (int)vector.get_size());
    checkCudaStatus(cudaStreamSynchronize(vector.get_stream()));
    log("\n");
}

template <typename Type> void stdout_vector(Type *vector, size_t size, const char *title, cudaStream_t stream) {
    log("%s --- --- ---\n", title);

    stdout_vector_kernel<Type>(stream, vector, size);
    checkCudaStatus(cudaStreamSynchronize(stream));
    log("\n");
}

template <typename Type> void stdout_vector(host_vector<Type> &vector, const char *title) {
    const int size = vector.get_size();
    const Type *data = vector.data();
    printer<Type> printer;
    log("%s --- --- --- \n", title);
    int i = size;
    while (i-- > 0) {
        printer(i, data[i]);
    }
    log("\n");
}

/// ===================================================== ===================================================== ///

} // namespace utility

#endif // _QUDA_CUH_