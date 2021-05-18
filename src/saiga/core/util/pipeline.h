/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/util/Thread/SynchronizedBuffer.h"
#include "saiga/core/util/Thread/threadName.h"

#include <mutex>
#include <thread>

#include <condition_variable>

namespace Saiga
{
/**
 * Class to create Pipeline-parallel algorithms.
 */
template <typename OutputType, int queueSize = 1, bool override = true>
class SAIGA_TEMPLATE PipelineStage
{
   public:
    PipelineStage(const std::string& name = "PipelineStage") : buffer(queueSize), name(name) {}
    ~PipelineStage() { stop(); }

    template <typename T>
    void run(T op)
    {
        running = true;
        // Capture 'op' by copy because it runs out of scope.
        // Note: 'this' is still captured by reference
        t = std::thread([&, op]() {
            setThreadName(name);
            OutputType tmp;
            while (running)
            {
                tmp = op();

                // Do not add objects that convert to false.
                if (!tmp) continue;

                if (override)
                {
                    buffer.addOverride(tmp);
                }
                else
                {
                    buffer.add(tmp);
                }
            }
        });
    }

    void stop()
    {
        if (running)
        {
            running = false;
            if (t.joinable()) t.join();
        }
    }

    void get(OutputType& out) { out = buffer.get(); }

    bool tryGet(OutputType& out) { return buffer.tryGet(out); }

    std::string getName() const { return name; }

   private:
    volatile bool running = false;
    SynchronizedBuffer<OutputType> buffer;
    std::thread t;
    std::string name;
};

}  // namespace Saiga
