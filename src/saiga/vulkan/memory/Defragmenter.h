//
// Created by Peter Eichinger on 2019-01-21.
//

#pragma once

#include "saiga/util/easylogging++.h"

#include <atomic>
#include <mutex>
#include <thread>

#include <condition_variable>

class Defragmenter
{
   private:
    bool start_defrag, running;
    std::mutex find_mutex;
    std::condition_variable find_condition;

   public:
    Defragmenter()
    {
        std::thread worker(&Defragmenter::worker_func, this);
        {
            std::lock_guard<std::mutex> lock(find_mutex);
            start_defrag = true;
        }
        find_condition.notify_one();
    }

    void pause() { running = false; }

    void worker_func()
    {
        std::unique_lock<std::mutex> lock(find_mutex);
        find_condition.wait(lock, [this] { return start_defrag; });

        // while (start_defrag)
        //{
        while (running)
        {
            LOG(INFO) << "Running";
        }
        //}
    }
};
