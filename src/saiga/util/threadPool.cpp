//Copyright (c) 2012 Jakob Progsch, Václav Zeman

//This software is provided 'as-is', without any express or implied
//warranty. In no event will the authors be held liable for any damages
//arising from the use of this software.

//Permission is granted to anyone to use this software for any purpose,
//including commercial applications, and to alter it and redistribute it
//freely, subject to the following restrictions:

//   1. The origin of this software must not be misrepresented; you must not
//   claim that you wrote the original software. If you use this software
//   in a product, an acknowledgment in the product documentation would be
//   appreciated but is not required.

//   2. Altered source versions must be plainly marked as such, and must not be
//   misrepresented as being the original software.

//   3. This notice may not be removed or altered from any source
//distribution.

/**
  * This file was modified by Darius Rueckert for libsaiga.
  */

#include "threadPool.h"
#include "saiga/util/threadName.h"

namespace Saiga {




ThreadPool::ThreadPool(size_t threads)
    :   stop(false)
{
    workingThreads = threads;
    for(size_t i = 0; i < threads; ++i)
    {
        workers.emplace_back(
                    [this,i]
        {
            setThreadName("ThreadPool Thread " + std::to_string(i));
            for(;;)
            {
                std::function<void()> task;

                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    workingThreads--;
                    this->condition.wait(lock,
                                         [this]{ return this->stop || !this->tasks.empty(); });
                    if(this->stop && this->tasks.empty())
                        return;
                    workingThreads++;
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }

                task();
            }
        }
        );
    }
}

ThreadPool::~ThreadPool()
{
    quit();
}

void ThreadPool::quit()
{
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if(stop)
            return;
        stop = true;
    }
    condition.notify_all();
    for(std::thread &worker: workers)
        worker.join();
    workers.clear();
}

std::shared_ptr<ThreadPool> globalThreadPool;

void createGlobalThreadPool(int threads)
{
    globalThreadPool = std::make_shared<ThreadPool>(threads);
}



}
