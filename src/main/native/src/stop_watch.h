#ifndef _STOP_WATCH_H_
#define _STOP_WATCH_H_

#include <chrono>



/// header-only

namespace graph
{

struct ClockNano
{
    long operator()()
    {
        auto epoch = std::chrono::high_resolution_clock::now().time_since_epoch();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(epoch).count();
    }
};

struct ClockMillis
{
    long operator()()
    {
        auto epoch = std::chrono::system_clock::now().time_since_epoch();
        return std::chrono::duration_cast<std::chrono::milliseconds>(epoch).count();
    }
};

template <typename ClockFactory = ClockMillis> class StopWatch
{
    ClockFactory clock;
  public:
    StopWatch()
    {
        reset();
    }

    void reset()
    {
        startTick = 0L;
        stopTick = 0L;
        accTime = 0L;
    }

    void setStartTick()
    {
        startTick = clock();
    }

    void setStopTick()
    {
        stopTick = clock();
        accTime += (stopTick - startTick);
    }

    long delta()
    {
        return stopTick - startTick;
    }

    long startTick;
    long stopTick;
    long accTime;
};

} // namespace graph

#endif //_STOP_WATCH_H_
