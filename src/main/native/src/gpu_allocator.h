#ifndef _CU_ALLOCATOR_H_
#define _CU_ALLOCATOR_H_

#include "cuda_runtime.h"
#include "stdio.h"

#include <memory>


/// <summary>
/// Cuda fast pinned memory allocator as required by std::vector
/// </summary>
/// <typeparam name="_Ty"></typeparam>
template <class _Ty> class gpu_allocator {
  public:
    static_assert(!std::is_const_v<_Ty>, "The C++ Standard forbids containers of const elements "
                                         "because allocator<const T> is ill-formed.");
    using _From_primary = gpu_allocator;

    using value_type = _Ty;

    using size_type = size_t;
    using difference_type = ptrdiff_t;

    using propagate_on_container_move_assignment = std::true_type;

    constexpr gpu_allocator() noexcept {}

    constexpr gpu_allocator(const gpu_allocator &) noexcept = default;

    template <class _Other> constexpr gpu_allocator(const gpu_allocator<_Other> &) noexcept {}

    _CONSTEXPR20 ~gpu_allocator() = default;

    _CONSTEXPR20 gpu_allocator &operator=(const gpu_allocator &) = default;

    _CONSTEXPR20 void deallocate(_Ty *const _Ptr, const size_t _Count) {
        cudaError_t status = cudaFreeHost(_Ptr);
        if (status != cudaSuccess) {
            fprintf(stderr, "[cuda/pinned] deallocation failed !\n");
            const char *error_name = cudaGetErrorName(status);
            const char *error_str = cudaGetErrorString(status);
            fprintf(stderr, " cuda deallocation of pinned memory failed ; ( %s )  %s \n", error_name, error_str);
        }
    }

    /// custom VS memory allocation trace events - __declspec(allocator) - compiler adapters
    /// https://docs.microsoft.com/en-us/cpp/cpp/declspec?view=msvc-170
    ///
    _NODISCARD _CONSTEXPR20 __declspec(allocator) _Ty *allocate(_CRT_GUARDOVERFLOW const size_t _Count) {
        _Ty *ptr = nullptr;
        size_t size = std::_Get_size_of_n<sizeof(_Ty)>(_Count);
        cudaError_t status = cudaHostAlloc((void **)&ptr, size, 0);
        if (status != cudaSuccess) {
            fprintf(stderr, "[cuda/pinned] deallocation failed !\n");
            const char *error_name = cudaGetErrorName(status);
            const char *error_str = cudaGetErrorString(status);
            fprintf(stderr, " cuda allocation of pinned memory failed ; ( %s )  %s \n", error_name, error_str);
            exit(1);
        }
        return ptr;
    }
};



/// <summary>
/// register additional operator==  in  std namespace
/// </summary>
namespace std {

template <class _Ty, class _Other>
_NODISCARD _CONSTEXPR20 bool operator==(const gpu_allocator<_Ty> &, const gpu_allocator<_Other> &) noexcept {
    return true;
}

#endif // _CU_ALLOCATOR_H_

