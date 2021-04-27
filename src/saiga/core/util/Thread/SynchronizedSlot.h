/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <atomic>
#include <mutex>

#include <condition_variable>

namespace Saiga
{
template <typename T>
class SynchronizedSlot
{
   public:
    bool empty() { return !valid; }
    bool full() { return valid; }

    template <typename G>
    void set(G&& value)
    {
        std::unique_lock l(mut);
        // Wait until empty
        cv_producer.wait(l, [this]() { return empty(); });
        // Set slot and notify the consumer
        slot  = std::move(value);
        valid = true;
        cv_consumer.notify_one();
    }

    template <typename G>
    void setOverride(G&& value)
    {
        std::unique_lock l(mut);
        // Set slot and notify the consumer
        // Do not wait if already full
        slot  = std::move(value);
        valid = true;
        cv_consumer.notify_one();
    }

    T get()
    {
        std::unique_lock l(mut);
        // Wait until full
        cv_consumer.wait(l, [this]() { return full(); });
        cv_producer.notify_one();
        valid = false;
        return std::move(slot);
    }

   private:
    std::atomic<bool> valid = false;
    std::mutex mut;
    std::condition_variable cv_producer;
    std::condition_variable cv_consumer;
    T slot;
};


}  // namespace Saiga
