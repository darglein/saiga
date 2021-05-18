/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/DataStructures/managedBuffer.h"


namespace Saiga
{
#if 0
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
//        return count() == (int)capacity;
        return front == rear;
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

    //adds one element to the buffer
    //overrides the first element if full
    void addOverride(const T& data)
    {
        if(full())
        {
            // Override first element and increment both pointers
            (*this)[front] = data;
            front = (front + 1) % capacity;
            rear = (rear + 1) % capacity;
        }else
        {
            add(data);
        }
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

    void clear()
    {
        front = -1;
        rear = 0;
        this->freeAll();
    }
};

#else
template <typename T>
class SAIGA_TEMPLATE RingBuffer : public std::vector<T>
{
   public:
    // pointer to the first element. -1 means there is no element
    int front = -1;
    // pointer to the first free spot at the end
    // if rear==front then this buffer is full
    int rear = 0;



    int _capacity;

    RingBuffer(int capacity) : std::vector<T>(capacity), _capacity(capacity) {}

    ~RingBuffer() {}

    bool empty() const { return front == -1; }

    int capacity() const { return _capacity; }

    bool full() const
    {
        //        return count() == (int)capacity;
        return front == rear;
    }

    int count() const
    {
        if (empty()) return 0;
        return (front < rear) ? rear - front : rear + capacity() - front;
    }


    // adds one element to the buffer
    template <typename G>
    void add(G&& data)
    {
        SAIGA_ASSERT(!full());
        if (empty()) front = rear;
        (*this)[rear] = std::forward<G>(data);
        rear          = (rear + 1) % capacity();
    }

    // adds one element to the buffer
    // if full
    //    - overrides the first element
    //    - increment the front point
    // Returns true if an element was actually overriden
    template <typename G>
    bool addOverride(G&& data)
    {
        if (full())
        {
            // Override first element and increment both pointers
            (*this)[front] = std::forward<G>(data);
            front          = (front + 1) % capacity();
            rear           = (rear + 1) % capacity();
            return true;
        }
        else
        {
            add(std::forward<G>(data));
            return false;
        }
    }

    // removes one element and returns it
    T get()
    {
        T result       = std::move((*this)[front]);
        (*this)[front] = T();  // override with default element
        front          = (front + 1) % capacity();
        if (front == rear) front = -1;
        return result;
    }


    void clear()
    {
        front = -1;
        rear  = 0;
        for (int i = 0; i < capacity(); ++i)
        {
            (*this)[i] = T();
        }
    }
};
#endif

}  // namespace Saiga
