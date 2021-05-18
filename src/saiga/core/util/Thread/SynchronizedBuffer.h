/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/DataStructures/ringBuffer.h"

#include <mutex>

#include <condition_variable>

namespace Saiga
{
template <typename T>
class SAIGA_TEMPLATE SynchronizedBuffer : protected RingBuffer<T>
{
   public:
    using Base = RingBuffer<T>;

    std::mutex lock;

    std::condition_variable not_full;
    std::condition_variable not_empty;

    SynchronizedBuffer(int capacity) : RingBuffer<T>(capacity) {}

    ~SynchronizedBuffer() {}

    int count()
    {
        std::unique_lock<decltype(lock)> l(lock);
        return Base::count();
    }

    int capacity() { return Base::capacity(); }
    bool emptysync()
    {
        std::unique_lock<decltype(lock)> l(lock);
        return this->front == -1;
    }

    void clear()
    {
        std::unique_lock<decltype(lock)> l(lock);
        Base::clear();
    }

    // blocks until buffer is empty
    void waitUntilEmpty()
    {
        std::unique_lock<decltype(lock)> l(lock);
        not_full.wait(l, [this]() { return this->empty(); });
    }

    void waitUntilFull()
    {
        std::unique_lock<decltype(lock)> l(lock);
        not_empty.wait(l, [this]() { return this->full(); });
    }



    template <typename G>
    void add(G&& data)
    {
        std::unique_lock<decltype(lock)> l(lock);
        not_full.wait(l, [this]() { return !this->full(); });
        RingBuffer<T>::add(std::forward<G>(data));
        not_empty.notify_one();
    }

    template <typename G>
    bool addOverride(G&& data)
    {
        std::unique_lock<decltype(lock)> l(lock);
        auto ret = RingBuffer<T>::addOverride(std::forward<G>(data));
        not_empty.notify_one();
        return ret;
    }

    bool tryAdd(const T& v)
    {
        std::unique_lock<decltype(lock)> l(lock);
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
        std::unique_lock<decltype(lock)> l(lock);
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
        std::unique_lock<decltype(lock)> l(lock);
        bool got_something = not_empty.wait_for(l, duration, [this]() { return !this->empty(); });
        if (!got_something) return T();
        T result = RingBuffer<T>::get();
        not_full.notify_one();
        return result;
    }

    bool tryGet(T& v)
    {
        std::unique_lock<decltype(lock)> l(lock);
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
