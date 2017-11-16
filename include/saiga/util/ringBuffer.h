/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include <vector>

namespace Saiga {

template<typename T>
class SAIGA_TEMPLATE RingBuffer{
public:
    std::vector<T> buffer;

    int capacity; //maximum capacity

    int front = 0; //pointer to the first element
    int rear = 0; //pointer to the first free spot at the end

    RingBuffer(int capacity) : buffer(capacity),capacity(capacity) {
    }

    ~RingBuffer(){
    }

	bool empty() const{
        return front == rear;
    }

	bool full() const{
        return count()>= capacity-1;
    }

    int count() const{
        return (front <= rear) ? rear-front : rear + capacity - front;
    }

	//adds one element to the buffer
    void add(const T& data)
    {
        buffer[rear] = data;
        rear = (rear + 1) % capacity;
    }

    //adds the element by swapping
    void addSwap(T& data)
    {
        swap(buffer[rear],data);
        rear = (rear + 1) % capacity;
    }

	//removes one element and returns it
    T get(){
        T result = buffer[front];
        front = (front + 1) % capacity;
        return result;
    }

    //removes one element and returns it
    bool getSwap(T& data){
        if(empty())
            return false;
        swap(buffer[front],data);
        front = (front + 1) % capacity;
        return true;
    }



};

}
