#ifndef _KERNEL_TRAITS_H_
#define _KERNEL_TRAITS_H_


/// <summary>
/// Kernel Dynamic Computation Characteristisc
/// </summary>
template <size_t OBJECT_PER_THREAD = 1, size_t T_DIM_BLOCK = 512> class KernelTraits {

  public:
    const unsigned _OBJECT_PER_THREAD = OBJECT_PER_THREAD;

    /// Expected block dimension default value
    const unsigned BLOCK_DIM_DEFAULT = T_DIM_BLOCK;

    /// total number of elements to process in kernel execution
    const unsigned container_size;

    /// align-up on elements per thread variable
    const unsigned aligned_container_size =
        ((unsigned)container_size + _OBJECT_PER_THREAD - 1) / _OBJECT_PER_THREAD * _OBJECT_PER_THREAD;

    /// absolut number of kernel threads in execution process
    const unsigned block_underflow = aligned_container_size / _OBJECT_PER_THREAD;

    /// Computed Grid Dimension
    const unsigned GRID_DIM = (block_underflow + BLOCK_DIM_DEFAULT - 1) / BLOCK_DIM_DEFAULT;

    /// Computed Block Dimension
    const unsigned BLOCK_DIM = (GRID_DIM == 1) ? block_underflow : BLOCK_DIM_DEFAULT;

    KernelTraits(unsigned int _container_size) : container_size(_container_size) {}

    KernelTraits(size_t _container_size) : KernelTraits(static_cast<unsigned int>(_container_size)) {}

};

#endif // _KERNEL_TRAITS_H_