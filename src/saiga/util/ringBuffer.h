/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/util/managedBuffer.h"


namespace Saiga {

template<typename T>
class SAIGA_TEMPLATE RingBuffer : public ManagedBuffer<T>
{
public:
    using ManagedBuffer<T>::capacity;
    using ManagedBuffer<T>::free;
    //pointer to the first element. -1 means there is no element
    int front = -1;
    //pointer to the first free spot at the end
    // if rear==front then this buffer is full
    int rear = 0;




    RingBuffer(int capacity)
        : ManagedBuffer<T>(capacity)
    {

    }

    ~RingBuffer(){
    }

    bool empty() const{
        return front == -1;
    }

    bool full() const{
        return count() == (int)capacity;
    }

    int count() const{
        if(empty()) return 0;
        return (front < rear) ? rear-front : rear + capacity - front;
    }


    //adds one element to the buffer
    void add(const T& data)
    {
        SAIGA_ASSERT(!full());
        if(empty()) front = rear;
        (*this)[rear] = data;
        rear = (rear + 1) % capacity;
    }

    //adds the element by swapping
    void addSwap(T& data)
    {
        if(empty()) front = rear;
        swap((*this)[rear],data);
        rear = (rear + 1) % capacity;
    }

    //removes one element and returns it
    T get()
    {
        T result = (*this)[front];
        free(front);
        front = (front + 1) % capacity;
        if(front == rear) front = -1;
        return result;
    }

    //removes one element and returns it
    bool getSwap(T& data)
    {
        if(empty())
            return false;
        swap((*this)[front],data);
        free(front);
        front = (front + 1) % capacity;
        if(front == rear) front = -1;
        return true;
    }



};

}
