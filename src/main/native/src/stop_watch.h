#ifndef _STOP_WATCH_H_
#define _STOP_WATCH_H_

#include <chrono>

namespace graph {

long TimeNanosecondsNow();

class StopWatch {
      public:
        StopWatch();

        void setStartTick();

        void setStopTick();

        long delta();

        void reset();

        long startTick;
        long stopTick;
        long accTime;
};

} // namespace graph

#endif //_STOP_WATCH_H_