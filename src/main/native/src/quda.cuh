#ifndef _QUDA_CUH_
#define _QUDA_CUH_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>

#include "cuerror.h"

#include "gpu_allocator.h"


} // namespace std

namespace quda {

template <typename Type> class dev_vector;

/// ===================================================== ===================================================== ///

/// <summary>
/// std vector with characteristics to supply additional memmory from cuda pinned memory pool
/// </summary>
template <typename Type> using cu_vector = std::vector<Type, gpu_allocator<Type>>;

/// ===================================================== ===================================================== ///

template <typename Type> class host_vector {
  public:
    /// <summary>
    /// allocate owned pinned memory extent for requested type and size
    /// </summary>
    /// <param name="size"></param>
    host_vector(size_t size) : _size(size), owner(true) {
        cudaError_t status = cudaMallocHost((void **)&_memory, size * sizeof(Type));
        if (status != cudaSuccess) {
            fprintf(stderr, "[host_vector] vector memory allocation failed !\n");
            exit(1);
        }
    }

    /// <summary>
    /// initialize non owned reference to host memory. implicit conversion
    /// </summary>
    /// <param name="source"></param>
    host_vector(const std::vector<Type> &source) : host_vector(source.data(), source.size()) {}

    /// <summary>
    /// initialize now owned reference to host memory.
    /// </summary>
    /// <param name="memory"></param>
    /// <param name="size"></param>
    host_vector(Type *memory, size_t size) : owner(false), _memory(memory), _size(size) {}

    /// <summary>
    /// default accesor to memory reference
    /// </summary>
    /// <returns></returns>
    operator Type *() { return data(); }

    /// <summary>
    /// default accesor to memory reference
    /// </summary>
    /// <returns></returns>
    operator void *() { return data(); }

    Type *data() { return _memory; }

    /// <summary>
    /// copy host vector possible initialized with cuda pinned memory.
    /// </summary>
    /// <param name="src"></param>
    /// <param name="stream"></param>
    /// <returns></returns>
    host_vector<Type> &memcpy_of(host_vector<Type> &src, cudaStream_t stream = nullptr) {
        if (_size != src.size()) {
            fprintf(stderr, "[dev_vector] allocation buffers on device and host are not identical !\n");
            exit(1);
        }
        cudaError_t status;
        if (stream) {
            status = cudaMemcpyAsync(_memory, src.data(), sizeof(Type) * _size, cudaMemcpyHostToHost, stream);
        } else {
            status = cudaMemcpy(_memory, src.data(), sizeof(Type) * _size, cudaMemcpyHostToHost);
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
    /// copy device vector on the selected cuda stream asynchronously.
    /// </summary>
    /// <param name="src"></param>
    /// <param name="stream"></param>
    /// <returns></returns>
    host_vector<Type> &memcpy_of(dev_vector<Type> &src, cudaStream_t stream = nullptr) {
        if (_size != src.size()) {
            fprintf(stderr, "[dev_vector] allocation buffers on device and host are not identical !\n");
            exit(1);
        }
        cudaError_t status;
        if (stream) {
            status = cudaMemcpyAsync(_memory, src.data(), sizeof(Type) * _size, cudaMemcpyDeviceToHost, stream);
        } else {
            status = cudaMemcpy(_memory, src.data(), sizeof(Type) * _size, cudaMemcpyDeviceToHost);
        }
        if (status != cudaSuccess) {
            const char *error_name = cudaGetErrorName(status);
            const char *error_str = cudaGetErrorString(status);
            fprintf(stderr, " [host_vector] memcpy from device failed !( %s )  %s \n", error_name, error_str);
            exit(1);
        }
        return *this;
    }

    size_t size() const { return _size; }

    /// release memory into cuda driver only when in possession of this memory extent
    ~host_vector() {
        if (!owner)
            return;
        if (_memory) {
            cudaError_t status = cudaFreeHost(_memory);
            if (status != cudaSuccess) {
                const char *errorName = cudaGetErrorName(status);
                const char *errorStr = cudaGetErrorString(status);
                fprintf(stderr, "[host_vector] memory deallocation failed !\n", errorName, errorStr);
            }
        }
    }

  private:
    Type *_memory;
    size_t _size;
    bool owner;
};

/// ===================================================== ===================================================== ///
template <typename Type> class dev_vector {
  public:
    dev_vector(dev_vector const &other) = delete;

    /// <summary>
    /// setup non ownig reference to device memory extent
    /// </summary>
    /// <param name="memory"></param>
    /// <param name="size"></param>
    /// <param name="stream"></param>
    dev_vector(Type *memory, size_t size, cudaStream_t stream)
        : _memory(memory), _size(size), stream(stream), owner(false) {}

    /// <summary>
    /// initialize owning memory extent possibly allocated on selected stream
    /// </summary>
    /// <param name="size"></param>
    /// <param name="stream"></param>
    dev_vector(size_t size, cudaStream_t stream = nullptr) : _size(size), stream(stream), owner(true) {
        cudaError_t status;
        if (stream) {
            status = cudaMallocAsync((void **)&_memory, size * sizeof(Type), stream);
        } else {
            status = cudaMalloc((void **)&_memory, size * sizeof(Type));
        }
        if (status != cudaSuccess) {
            const char *errorName = cudaGetErrorName(status);
            const char *errorStr = cudaGetErrorString(status);
            fprintf(stderr, "[dev_vector] vector memory allocation failed ; ( %s )  %s \n", errorName, errorStr);
            exit(1);
        }
    }

    /// <summary>
    /// default accesor to memory reference
    /// </summary>
    /// <returns></returns>
    operator Type *() { return data(); }

    /// <summary>
    /// default accesor to memory reference
    /// </summary>
    /// <returns></returns>
    operator void *() { return data(); }

    Type *data() { return _memory; }

    size_t size() const { return _size; }

    cudaStream_t get_stream() { return stream; }

    /// <summary>
    /// copy host vector on selected cuda stream asynchronously into device.
    /// </summary>
    /// <param name="src"></param>
    /// <param name="stream"></param>
    /// <returns></returns>
    dev_vector<Type> &memcpy_of(host_vector<Type> &src, cudaStream_t stream = nullptr) {
        if (_size != src.size()) {
            fprintf(stderr, "[dev_vector] allocation buffers on device and host are not identical !\n");
            exit(1);
        }
        cudaError_t status;
        if (stream) {
            status = cudaMemcpyAsync(_memory, src.data(), sizeof(Type) * _size, cudaMemcpyHostToDevice, stream);
        } else {
            status = cudaMemcpy(_memory, src.data(), sizeof(Type) * _size, cudaMemcpyHostToDevice);
        }
        if (status != cudaSuccess) {
            const char *errorName = cudaGetErrorName(status);
            const char *errorStr = cudaGetErrorString(status);
            fprintf(stderr, "[dev_vector] memcpy into device failed ; ( %s )  %s \n", errorName, errorStr);
            exit(1);
        }
        return *this;
    }

    /// <summary>
    /// copy device vector on selected cuda stream asynchronously into device.
    /// </summary>
    /// <param name="src"></param>
    /// <param name="stream"></param>
    /// <returns></returns>
    dev_vector<Type> &memcpy_of(dev_vector<Type> &src, cudaStream_t stream = nullptr) {
        if (_size != src.size()) {
            fprintf(stderr, "[dev_vector] device allocations are not identical ! \n");
            exit(1);
        }
        cudaError_t status;
        if (stream) {
            status = cudaMemcpyAsync(_memory, src.data(), sizeof(Type) * _size, cudaMemcpyDeviceToDevice, stream);
        } else {
            status = cudaMemcpy(_memory, src.data(), sizeof(Type) * _size, cudaMemcpyDeviceToDevice);
        }
        if (status != cudaSuccess) {
            const char *errorName = cudaGetErrorName(status);
            const char *errorStr = cudaGetErrorString(status);
            fprintf(stderr, "[dev_vector] memcpy on device failed ; ( %s )  %s \n", errorName, errorStr);
            exit(1);
        }
        return *this;
    }

    /// release memory into cuda driver only when in possession of this memory extent
    ~dev_vector() {
        if (!owner) {
            return;
        }
        if (_memory) {

            cudaError_t status;
            if (stream) {
                status = cudaFreeAsync(_memory, stream);
            } else {
                status = cudaFree(_memory);
            }
            if (status != cudaSuccess) {
                const char *errorName = cudaGetErrorName(status);
                const char *errorStr = cudaGetErrorString(status);
                fprintf(stderr, "[dev_vector] memory deallocation failed ! ( %s ) %s \n", errorName, errorStr);
            }
        }
    }

  private:
    /// stream of orgin for this memory extent  - malloc/free
    cudaStream_t stream;

    /// memory extent in possession or borrowed
    Type *_memory;

    size_t _size;

    bool owner;
};
/// ===================================================== ===================================================== ///

template <typename... Type> __device__ __host__ void log(const char *format, Type &...values) {
    ::printf(format, values...);
}

/// ===================================================== ===================================================== ///

template <typename Type> __device__ __host__ const char *line_format();

template <typename Type> void stdout_vector_kernel(Type *vector, int size);

template <typename Type> void stdout_vector(dev_vector<Type> &vector, const char *title) {
    log("%s --- --- ---\n", title);

    stdout_vector_kernel<Type>(vector.data(), vector.size());
    checkCudaStatus(cudaStreamSynchronize(0));
    log("\n");
}

template <typename Type> void stdout_vector(host_vector<Type> &vector, const char *title) {
    const int size = vector.size();
    const Type *array = vector.data();
    const char *format = line_format<Type>();
    log("%s --- --- --- \n", title);
    int i = size;
    while (i-- > 0) {
        log(format, i, array[i]);
    }
    log("\n");
}

/// ===================================================== ===================================================== ///

} // namespace quda

#endif // _QUDA_CUH_