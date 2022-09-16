#ifndef _QUDA_CUH_
#define _QUDA_CUH_

#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>

#include "cuerror.h"

#include "gpu_allocator.h"
#include "model.cuh"

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
    dev_vector() : dev_vector(nullptr, 0, nullptr) {}

    /// <summary>
    /// Take a snapshot of reference device vector.
    /// </summary>
    /// <param name="other"></param>
    dev_vector(dev_vector const &other) noexcept : dev_vector(){ *this = other;
    }; 

    /// <summary>
    /// Become a new owner of other memory allocation.
    /// </summary>
    /// <param name="other"></param>
    dev_vector(dev_vector &&other) noexcept : dev_vector() { *this = std::move(other); };

    /// <summary>
    /// setup non ownig reference to device memory extent
    /// </summary>
    /// <param name="memory"></param>
    /// <param name="size"></param>
    /// <param name="stream"></param>
    dev_vector(Type *memory, size_t size, cudaStream_t stream = nullptr) : memory(memory), size(size), stream(stream), owner(false) {}

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
    dev_vector &operator=(dev_vector &&other) noexcept {
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
    dev_vector &operator=(dev_vector const &other) noexcept {
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

template <typename... Type> __device__ __host__ void infoLog(const char *format, Type const &...values) { ::printf(format, values...); }

/// ===================================================== ===================================================== ///

constexpr const char LINE_SEPARATOR[] = {"\n"};
constexpr const char VECTOR_HEADER[] = {"%s [ %d ] --- --- --- \n"};

template <typename Type> void printer(int i, Type object);

template <typename Type> void stdout_vector(host_vector<Type> &vector, const char *title) {
    const size_t size = vector.get_size();
    const Type *data = vector.data();
    infoLog(VECTOR_HEADER, title, size);
    int i = -1;
    while (++i <size) {
        printer<Type>(i, data[i]);
    }
    infoLog(LINE_SEPARATOR);
}

template <typename Type> void stdout_vector(dev_vector<Type> &vector, const char *title) {
    cudaStream_t stream = vector.get_stream();
    utility::host_vector<Type> host_view(vector.get_size());

    host_view.memcpy_of(vector, stream);
    if (stream) {
        checkCudaStatus(cudaStreamSynchronize(stream));
    }
    stdout_vector<Type>(host_view, title);
}

/// ===================================================== ===================================================== ///

/// =======================================================================================
///                             debug utility
/// =======================================================================================

constexpr const char *WIDEN_DOUBLE_STR_FORMAT = "%26d";
constexpr const char *FORMAT_STR_DOUBLE = " %11.2e";
constexpr const char *FORMAT_STR_IDX_DOUBLE = "%% %2d  %11.2e \n";
constexpr const char *FORMAT_STR_IDX_DOUBLE_E = "%%     %11.2e \n";

constexpr const char *FORMAT_STR_IDX_INT = "%% %2d  %2d \n";
constexpr const char *FORMAT_STR_IDX_INT_E = "%%     %2d \n";

constexpr const char *FORMAT_STR_DOUBLE_CM = ", %11.2e";

template <typename Type> void step_printer(int idx, Type value);

/// =======================================================================================
/// <summary>
/// [debug] State Vector print from device
/// </summary>

template <typename Type> void stdoutDeviceVector(Type *d_vector, size_t dimension, cudaStream_t stream, const char *title) {
    utility::host_vector<Type> stateVector{dimension};
    stateVector.memcpy_of(utility::dev_vector<Type>(d_vector, dimension, stream));
    if (stream) {
        checkCudaStatus(cudaStreamSynchronize(stream));
    }
    infoLog(LINE_SEPARATOR);
    infoLog(title);
    infoLog("\n MatrixDouble1 - %lu x 1 ****************************************\n", dimension);
    infoLog(LINE_SEPARATOR);

    for (int i = 0; i < dimension; i++) {
        Type value = stateVector[i];
        step_printer<Type>(i, value);        
    }
    infoLog(LINE_SEPARATOR);
}

///  ======================================================================================
/// <summary>
/// [debug] stdout tensor A - dense form.
/// </summary>
/// <param name="ecdata"></param>
/// <param name="rows"></param>
/// <param name="cols"></param>
/// <returns></returns>
void stdoutTensorData(double *d_A, size_t ld, size_t cols, cudaStream_t stream, const char *title);


/// ===================================================== ===================================================== ///

} // namespace utility

#endif // _QUDA_CUH_