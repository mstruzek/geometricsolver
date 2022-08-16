#include "stop_watch.h"


namespace graph {


long TimeNanosecondsNow() {
        auto fromEpoch = std::chrono::high_resolution_clock::now().time_since_epoch();
        return std::chrono::duration<long, std::nano>(fromEpoch).count();
}


StopWatch::StopWatch() { reset(); }

void StopWatch::setStartTick() { startTick = TimeNanosecondsNow(); }

void StopWatch::setStopTick() {
        stopTick = TimeNanosecondsNow();
        accTime += (stopTick - startTick);
}

long StopWatch::delta() { return stopTick - startTick; }

void StopWatch::reset() {
        startTick = 0L;
        stopTick = 0L;
        accTime = 0L;
}

} // namespace graph