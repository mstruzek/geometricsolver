#ifndef _STOP_WATCH_H_
#define _STOP_WATCH_H_

#include <chrono>

namespace graph {

long TimeNanosecondsNow() {
        auto fromEpoch = std::chrono::high_resolution_clock::now().time_since_epoch();
        return std::chrono::duration<long, std::nano>(fromEpoch).count();
}

class StopWatch {
      public:
        StopWatch() { reset(); }

        void setStartTick() { startTick = TimeNanosecondsNow(); }

        void setStopTick() {
                stopTick = TimeNanosecondsNow();
                accTime += (stopTick - startTick);
        }

        long delta() { return stopTick - startTick; }

        void reset() {
                startTick = 0L;
                stopTick = 0L;
                accTime = 0L;
        }

        long startTick;
        long stopTick;
        long accTime;
};

} // namespace graph

#endif //_STOP_WATCH_H_