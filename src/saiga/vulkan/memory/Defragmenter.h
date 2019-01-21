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
    bool running, quit;
    std::mutex run_mutex;
    std::condition_variable run_cond;
    std::thread worker;

    void worker_func()
    {
        while (!quit)
        {
            std::unique_lock<std::mutex> lock(run_mutex);
            run_cond.wait(lock, [this] { return running || quit; });

            while (running)
            {
                LOG(INFO) << "Running";
            }
        }
    }

   public:
    Defragmenter() : running(false), quit(false), worker(&Defragmenter::worker_func, this) {}

    ~Defragmenter()
    {
        quit    = true;
        running = false;

        {
            std::lock_guard<std::mutex> lock(run_mutex);
        }
        run_cond.notify_one();
        worker.join();
    }

    void start()
    {
        if (running)
        {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(run_mutex);
            running = true;
        }
        run_cond.notify_one();
    }
    void stop() { running = false; }
};
