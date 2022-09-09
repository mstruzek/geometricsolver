#ifndef _QUDA_CUH_
#define _QUDA_CUH_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <stdio.h>

namespace quda {

template <typename Type> class dev_vector;

/// ===================================================== ===================================================== ///

template <typename Type> class host_vector {
  public:
    host_vector(Type *memory, size_t size) : owner(false), _memory(memory), _size(size) {}

    host_vector(size_t size) : _size(size), owner(true) {
        cudaError_t status = cudaMallocHost((void **)&_memory, size * sizeof(Type));
        if (status != cudaSuccess) {
            fprintf(stderr, "[host_vector] vector memory allocation failed !\n");
            exit(1);
        }
    }

    Type *operator*() { return data(); }

    Type *data() { return _memory; }

    host_vector<Type> &copy_of(dev_vector<Type> &src) {
        int source_size = src.size();
        if (_size != src_size) {
            fprintf(stderr, "[dev_vector] allocation buffers on device and host are not identical !\n");
            exit(1);
        }
        Type *source_mem = src.data();
        cudaStream_t source_stream = src.get_stream();

        cudaError_t status;
        if (source_stream) {
            status = cudaMemcpyAsync(_memory, source_mem, sizeof(Type) * _size, cudaMemcpyDeviceToHost, source_stream);
        } else {
            status = cudaMemcpy(_memory, source_mem, sizeof(Type) * _size, cudaMemcpyDeviceToHost);
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
            _memory = NULL;
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

    dev_vector(dev_vector &&other) { std::exchange(this, std::move(other)); }

    dev_vector &operator=(dev_vector &&other) {
        std::exchange(this, std::move(other));
        return *this;
    }

    dev_vector(Type *memory, size_t size, cudaStream_t stream)
        : _memory(memory), _size(size), stream(stream), owner(false) {}

    dev_vector(size_t size) : _size(size), stream(NULL), owner(true) {
        cudaError_t status = cudaMalloc((void **)&_memory, size * sizeof(Type));
        if (status != cudaSuccess) {
            const char *errorName = cudaGetErrorName(status);
            const char *errorStr = cudaGetErrorString(status);
            fprintf(stderr, "[dev_vector] vector memory allocation failed ; ( %s )  %s \n", errorName, errorStr);
            exit(1);
        }
    }

    dev_vector(size_t size, cudaStream_t stream) : _size(size), stream(stream), owner(true) {
        cudaError_t status = cudaMallocAsync((void **)&_memory, size * sizeof(Type), stream);
        if (status != cudaSuccess) {
            const char *errorName = cudaGetErrorName(status);
            const char *errorStr = cudaGetErrorString(status);
            fprintf(stderr, "[dev_vector] vector memory allocation failed ; ( %s )  %s \n", errorName, errorStr);
            exit(1);
        }
    }

    dev_vector<Type> &memcpy_of(host_vector<Type> &src) {
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

    dev_vector<Type> &copy_of(dev_vector<Type> &src) {
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

    Type *operator*() { return data(); }

    Type *data() { return _memory; }

    size_t size() const { return _size; }

    cudaStream_t get_stream() { return stream; }

    ~dev_vector() {
        if (!owner) {
            return;
        }
        if (_memory) {
            cudaError_t status = cudaFreeAsync(_memory, stream);
            if (status != cudaSuccess) {
                const char *errorName = cudaGetErrorName(status);
                const char *errorStr = cudaGetErrorString(status);
                fprintf(stderr, "[dev_vector] memory deallocation failed !\n");
            }
            _memory = NULL;
        }
    }

  private:
    cudaStream_t stream;
    Type *_memory;
    size_t _size;
    bool owner;
};

template <typename Type>
void dev_vector_memcpy(dev_vector<Type>& target , std::vector<Type> const &source, cudaStream_t stream) {
    target = dev_vector<Type>{source.size(), stream};
    host_vector<Type> host{source.data(), source.size()};
    target.memcpy_from(host);
}

/// ===================================================== ===================================================== ///

template <typename... Ty> __device__ __host__ void log(const char *format, Ty &...values) {
    ::printf(format, values...);
}

/// ===================================================== ===================================================== ///

template <typename Ty> __device__ __host__ const char *line_format();


template <typename Type> 
__global__ void stdout_vector_kernel(Type *vector, int size);



template <typename Type> void stdout_vector(dev_vector<Type> &vector, const char *title) {
    log("%s ###########\n", title);

    /// ----  ARG CAPTURING - SERIALIZABLE COPY_CTOR, DTOR !
    ///
    /// ----  ADVICE : basic types or references transport !
    stdout_vector_kernel<Type><<<1, 1>>>(vector.data(), vector.size());

    cudaStreamSynchronize(0);
    log("\n");
}

template <typename Type> void stdout_vector(host_vector<Type> &vector, const char *title) {
    const int size = vector.size();
    const Type *array = vector.data();
    const char *format = line_format<Type>();
    log("%s ###########\n", title);
    int i = size;
    while (i-- > 0) {
        log(format, i, array[i]);
    }
    log("\n");
}

/// ===================================================== ===================================================== ///

} // namespace quda

#endif // _QUDA_CUH_