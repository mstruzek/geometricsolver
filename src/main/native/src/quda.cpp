#include "quda.h"

#include <string>

#include "gpu_allocator.h"

namespace utility {


template <> void printer<int>(int i, int object) { printf("%d  %d \n", i, object); }

template <> void printer<double>(int i, double object) { printf("%d  %f\n", i, object); }


template <> void step_printer<double>(int idx, double value) {
    if (idx % 2 == 0) {
        infoLog(FORMAT_STR_IDX_DOUBLE, idx, value);
    } else {
        infoLog(FORMAT_STR_IDX_DOUBLE_E, value);
    }
}

template <> void step_printer<int>(int idx, int value) {
    if (idx % 2 == 0) {
        infoLog(FORMAT_STR_IDX_INT, idx, value);
    } else {
        infoLog(FORMAT_STR_IDX_INT_E, value);
    }
}


void stdoutTensorData(double *d_A, size_t ld, size_t cols, cudaStream_t stream, const char *title) {
    utility::host_vector<double> mA{ld * cols};
    mA.memcpy_of(utility::dev_vector<double>(d_A, ld * cols, stream));
    if (stream) {
        checkCudaStatus(cudaStreamSynchronize(stream));
    }
    infoLog(LINE_SEPARATOR);
    infoLog(title);
    infoLog("\n MatrixDouble2 - %lu x %lu **************************************** \n", ld, cols);
    infoLog(LINE_SEPARATOR);

    /// table header
    for (int i = 0; i < cols / 2; i++) {
        infoLog(WIDEN_DOUBLE_STR_FORMAT, i);
    }
    infoLog(LINE_SEPARATOR);

    /// table ecdata

    for (int i = 0; i < ld; i++) {
        infoLog(FORMAT_STR_DOUBLE, mA[ld * i]);

        for (int j = 1; j < cols; j++) {
            infoLog(FORMAT_STR_DOUBLE_CM, mA[ld * i + j]);
        }
        if (i < cols - 1)
            infoLog(LINE_SEPARATOR);
    }

    infoLog(LINE_SEPARATOR);
}


} // namespace utility