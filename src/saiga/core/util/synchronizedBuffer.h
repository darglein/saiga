/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/ringBuffer.h"

#include <mutex>

#include <condition_variable>

namespace Saiga
{
template <typename T>
class SAIGA_TEMPLATE SynchronizedBuffer : public RingBuffer<T>
{
   public:
    std::mutex lock;

    std::condition_variable not_full;
    std::condition_variable not_empty;

    SynchronizedBuffer(int capacity) : RingBuffer<T>(capacity) {}

    ~SynchronizedBuffer() {}

    // blocks until buffer is empty
    void waitUntilEmpty()
    {
        std::unique_lock<std::mutex> l(lock);
        not_full.wait(l, [this]() { return this->empty(); });
    }

    void waitUntilFull()
    {
        std::unique_lock<std::mutex> l(lock);
        not_empty.wait(l, [this]() { return this->full(); });
    }



    void add(const T& data)
    {
        std::unique_lock<std::mutex> l(lock);
        not_full.wait(l, [this]() { return !this->full(); });
        RingBuffer<T>::add(data);
        not_empty.notify_one();
    }

    void addOverride(const T& data)
    {
        std::unique_lock<std::mutex> l(lock);
        RingBuffer<T>::addOverride(data);
        not_empty.notify_one();
    }

    bool tryAdd(const T& v)
    {
        std::unique_lock<std::mutex> l(lock);
        if (this->full())
        {
            return false;
        }
        RingBuffer<T>::add(v);
        not_empty.notify_one();
        return true;
    }


    T get()
    {
        std::unique_lock<std::mutex> l(lock);
        not_empty.wait(l, [this]() { return !this->empty(); });
        T result = RingBuffer<T>::get();
        not_full.notify_one();
        return result;
    }

    // Blocks until we got an elemnt or the duration has passed.
    // Returns T() on timeout.
    template <typename TimeType>
    T getTimeout(const TimeType& duration)
    {
        std::unique_lock<std::mutex> l(lock);
        bool got_something = not_empty.wait_for(l, duration, [this]() { return !this->empty(); });
        if (!got_something) return T();
        T result = RingBuffer<T>::get();
        not_full.notify_one();
        return result;
    }

    bool tryGet(T& v)
    {
        std::unique_lock<std::mutex> l(lock);
        if (this->empty())
        {
            return false;
        }
        v = RingBuffer<T>::get();
        not_full.notify_one();
        return true;
    }
};

}  // namespace Saiga
