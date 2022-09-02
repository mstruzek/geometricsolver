#ifndef _STOP_WATCH_H_
#define _STOP_WATCH_H_

#include <chrono>

#include "settings.h"

/// header-only

namespace graph {

struct ClockNano {
    long long operator()() {
        auto epoch = std::chrono::high_resolution_clock::now().time_since_epoch();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(epoch).count();
    }
};

struct ClockMillis {
    long long operator()() {
        auto epoch = std::chrono::system_clock::now().time_since_epoch();
        return std::chrono::duration_cast<std::chrono::milliseconds>(epoch).count();
    }
};

template <typename ClockFactory = ClockMillis> class StopWatch {
    ClockFactory clock;

  public:
    StopWatch() { reset(); }

    void reset() {
        startTick = 0L;
        stopTick = 0L;
        accTime = 0L;
    }

    void setStartTick() { startTick = clock(); }

    void setStopTick() {
        stopTick = clock();
        accTime += (stopTick - startTick);
    }

    long long delta() const { return stopTick - startTick; }

    long long startTick;
    long long stopTick;
    long long accTime;
};

class StopWatchAdapter {

  public:
    StopWatchAdapter() {
        millis.reset();
        nanos.reset();
    }

    void setStartTick() {
        nanos.setStartTick();
        millis.setStartTick();
    }

    void setStopTick() {
        nanos.setStopTick();
        millis.setStopTick();
    }

    void reset() { 
        nanos.reset();
        millis.reset();
    }

    long long delta() {
        return (settings::get()->CLOCK_NANOSECONDS) ? nanos.delta() : millis.delta();
    }

    long long getStartTick() const {
        return (settings::get()->CLOCK_NANOSECONDS) ? nanos.startTick :  millis.startTick;
    }

    long long getStopTick() const { 
        return (settings::get()->CLOCK_NANOSECONDS) ? nanos.stopTick : millis.stopTick;
    }

  private:
    graph::StopWatch<graph::ClockNano> nanos;
    graph::StopWatch<graph::ClockMillis> millis;
    
};

} // namespace graph

#endif //_STOP_WATCH_H_
