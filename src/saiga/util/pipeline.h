/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include "saiga/util/threadName.h"
#include "saiga/util/synchronizedBuffer.h"

#include <condition_variable>
#include <mutex>
#include <thread>

namespace Saiga {

/**
 * Class to create Pipeline-parallel algorithms.
 */
template<typename OutputType, int queueSize = 1, bool override = true>
class SAIGA_TEMPLATE PipelineStage
{
public:
    PipelineStage(const std::string& name = "PipelineStage") : buffer(queueSize), name(name) {}
    ~PipelineStage()
    {
        if(running)
        {
            running = false;
            t.join();
        }
    }

    template<typename T>
    void run(T op)
    {
        running = true;
        t = std::thread(
                    [&]()
        {
            setThreadName(name);
            OutputType tmp;
            while(running)
            {
                tmp = op();
                if(override)
                {
                    buffer.addOverride(tmp);
                }else
                {
                    buffer.add(tmp);
                }
            }
        });
    }

    void get(OutputType& out)
    {
        out = buffer.get();
    }

    bool tryGet(OutputType& out)
    {
        return buffer.tryGet(out);
    }

    std::string getName() const { return name; }

private:
    volatile bool running = false;
    SynchronizedBuffer<OutputType> buffer;
    std::thread t;
    std::string name;
};

}

