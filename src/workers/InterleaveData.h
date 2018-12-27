
#ifndef XMRIG_INTERLEAVE_H
#define XMRIG_INTERLEAVE_H


#include <mutex>

struct InterleaveData
{
    std::mutex mutex;

    double adjustThreshold = 0.4;
    double startAdjustThreshold = 0.4;
    double avgKernelRuntime = 0.0;
    uint64_t lastRunTimeStamp = 0;
    uint32_t numThreadsOnGPU = 0;
};

#endif
