/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/util/ringBuffer.h"

#include <condition_variable>
#include <mutex>

namespace Saiga {

template<typename T>
class SAIGA_GLOBAL SynchronizedBuffer : public RingBuffer<T>{
public:
    std::mutex lock;

    std::condition_variable not_full;
    std::condition_variable not_empty;

    SynchronizedBuffer(int capacity) : RingBuffer<T>(capacity) {
    }

    ~SynchronizedBuffer(){
    }

    void add(T data){
        std::unique_lock<std::mutex> l(lock);
        not_full.wait(l, [this](){return !this->full();});
        RingBuffer<T>::add(data);
        not_empty.notify_one();
    }

    T get(){
        std::unique_lock<std::mutex> l(lock);
        not_empty.wait(l, [this](){return this->count() != 0; });
        T result = RingBuffer<T>::get();
        not_full.notify_one();
        return result;
    }

    bool tryGet(T& v){
        std::unique_lock<std::mutex> l(lock);
        if(this->empty()){
            return false;
        }

        v = RingBuffer<T>::get();
        not_full.notify_one();
        return true;
    }
};

}
