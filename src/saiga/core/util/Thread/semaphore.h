/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <mutex>

#include <condition_variable>

namespace Saiga
{
class SAIGA_CORE_API Semaphore
{
   public:
    Semaphore(int count_ = 0) : count(count_) {}

    /**
     * Increase counter and notify a thread that waits on the condition variable.
     */
    inline void notify()
    {
        std::unique_lock<std::mutex> lock(mtx);
        count++;
        cv.notify_one();
    }

    /**
     * Blocks until the counter is greater than 0.
     * Decrements the counter by one.
     */
    inline void wait()
    {
        std::unique_lock<std::mutex> lock(mtx);

        while (count == 0)
        {
            cv.wait(lock);
        }
        count--;
    }

    /**
     * Decrements the counter if it is larger than 1.
     * Returns immidietaly
     */
    inline bool trywait()
    {
        std::unique_lock<std::mutex> lock(mtx);
        if (count)
        {
            --count;
            return true;
        }
        else
        {
            return false;
        }
    }

   private:
    std::mutex mtx;
    std::condition_variable cv;
    int count;
};

}  // namespace Saiga
