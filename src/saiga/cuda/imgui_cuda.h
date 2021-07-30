/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/imgui/imgui_timer_system.h"
#include "saiga/cuda/cudaTimer.h"

namespace Saiga
{
namespace CUDA
{
class CudaTimerSystem : public TimerSystem
{
   public:
    CudaTimerSystem() : TimerSystem("CUDA Timer") {}

   protected:
    virtual void BeginFrameImpl() override { base_timer.startTimer(); }
    virtual void EndFrameImpl() override { base_timer.stopTimer(true); }

    virtual std::unique_ptr<TimestampTimer> CreateTimer() override
    {
        auto timer        = std::make_unique<CUDA::RelativeCudaTimer>();
        timer->base_timer = &base_timer;
        return timer;
    }

    CUDA::MultiFrameTimer base_timer;
};
}  // namespace CUDA
}  // namespace Saiga