
# DO NOT COMPILE 




template <size_t N>
void fillPointCoordinateVectorRound(const size_t offset, double *stateVector, std::vector<graph::Point>& points) {
#pragma unroll
    for (size_t i = 0; i < N; ++i) {
        const size_t ioff = offset + i;
        graph::Point &p = points[ioff];
        stateVector[2 * ioff] = p.x;
        stateVector[2 * ioff + 1] = p.y;
    }
}



/// NOT EFFIECIENT LOOP UNROLLING 

void GPUComputation::fillPointCoordinateVector(double *stateVector) {

    graph::StopWatchAdapter stopWatch;

    stopWatch.setStartTick();

#define ROUND_1 1
#define ROUND_2 2
#define ROUND_3 3
#define ROUND_4 1 << 2
#define ROUND_8 1 << 3
#define ROUND_16 1 << 4
#define ROUND_32 1 << 5
#define ROUND_64 1 << 6
#define ROUND_128 1 << 7
#define ROUND_256 1 << 8

#define BATCH_INIT (1 << 8)

    size_t W = 0; //_points.size();

    /// open iteration - warm-up
    size_t R = W;

    for (size_t BATCH = BATCH_INIT; BATCH > ROUND_2; BATCH >>= 1) {
        if (R == BATCH) {
            break;
        }
        int C = R / BATCH;
        if (C > 0) {
            R = R - C * BATCH;
        }
    }

    size_t offset = 0;
    while ((W - offset) > 0) {

/// ================================ ================================ ///
#define BATCH_ROUND(K)                                                                                                 \
    case K:                                                                                                            \
        /** fillPointCoordinateVectorRound<K>(offset, stateVector, _points);*/                                              \
        offset += K;                                                                                                   \
        break;

/// ================================ ================================ ///
        switch (R) {

        case 0:
            if ((W - offset) > 0) {
                /// KROTNOSCI
                R = W - offset;
                for (int i = 0, C = R / (ROUND_256); i < C; ++i) {
                    /* fillPointCoordinateVectorRound<ROUND_256>(offset, stateVector, _points);*/
                    offset += ROUND_256;
                }
            }
            goto Exit;
        break;
        BATCH_ROUND(ROUND_1)
        BATCH_ROUND(ROUND_2)
        BATCH_ROUND(ROUND_3)
        BATCH_ROUND(ROUND_4)
        BATCH_ROUND(ROUND_8)
        BATCH_ROUND(ROUND_16)
        BATCH_ROUND(ROUND_32)
        BATCH_ROUND(ROUND_64)
        BATCH_ROUND(ROUND_128)
        BATCH_ROUND(ROUND_256)
        }

        R = W - offset;

        // continue iteration on highest possible
        for (size_t BATCH = BATCH_INIT; BATCH > ROUND_2; BATCH >>= 1) {
            if (R == BATCH) {
                break;
            }
            int C = R / BATCH;
            if (C > 0) {
                R = R - C * BATCH;
            }
        }
    }

#undef ROUND_1
#undef ROUND_2
#undef ROUND_3
#undef ROUND_4
#undef ROUND_8
#undef ROUND_16
#undef ROUND_32
#undef ROUND_64
#undef ROUND_128
#undef ROUND_256
#undef BATCH_INIT
#undef BATCH_ROUND

Exit:
    stopWatch.setStopTick();

    fprintf(stdout, "fillPointCoordinateVector delta: %8.3e ", stopWatch.delta());
    fflush(stdout);
}
