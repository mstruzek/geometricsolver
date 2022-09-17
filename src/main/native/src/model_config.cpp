#include "model_config.h"


const char *getComputationName(ComputationMode computationMode) {
    static const char *COMPUTATION_MODES[4] = {
        "DENSE_MODE",
        "SPARSE_MODE",
        "DIRECT_MODE",
        "COMPACT_MODE",
    };
    return COMPUTATION_MODES[(int)computationMode];
}
