
#ifndef XMRIG_INTERLEAVE_H
#define XMRIG_INTERLEAVE_H


#include <mutex>

#include <hip/hip_runtime.h>

struct InterleaveData
{
    std::mutex mutex;

    double adjustThreshold = 0.4;
    double startAdjustThreshold = 0.4;
    double avgKernelRuntime = 0.0;
    uint64_t lastRunTimeStamp = 0;
    uint32_t numThreadsOnGPU = 0;

	hipEvent_t progress;
};

#endif
