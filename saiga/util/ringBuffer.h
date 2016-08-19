#pragma once

#include "saiga/config.h"
#include <vector>


template<typename T>
class SAIGA_GLOBAL RingBuffer{
public:
    std::vector<T> buffer;

    int capacity; //maximum capacity

    int front = 0; //pointer to the first element
    int rear = 0; //pointer to the first free spot at the end

    RingBuffer(int capacity) : buffer(capacity),capacity(capacity) {
    }

    ~RingBuffer(){
    }

    bool empty(){
        return front == rear;
    }

    bool full(){
        return count()>= capacity-1;
    }

    int count(){
        return (front <= rear) ? rear-front : rear + capacity - front;
    }

    void add(T data){
        buffer[rear] = data;
        rear = (rear + 1) % capacity;
    }

    T get(){
        T result = buffer[front];
        front = (front + 1) % capacity;
        return result;
    }
};
