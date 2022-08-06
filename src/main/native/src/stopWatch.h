#ifndef _STOP_WATCH_H_
#define _STOP_WATCH_H_

#include <chrono>

namespace gsketch {

long TimeNanosecondsNow() { return std::chrono::duration<long, std::nano>(std::chrono::high_resolution_clock::now().time_since_epoch()).count(); }

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

} // namespace gsketch

#endif //_STOP_WATCH_H_