#pragma once

#include "saiga/opengl/buffer.h"
#include <vector>

template<typename T>
class TemplatedBuffer : public Buffer{
public:

    TemplatedBuffer(GLenum _target): Buffer(_target){}
    ~TemplatedBuffer(){}

    void set(std::vector<T> &data, GLenum usage);
    void set(T* data, int count, GLenum usage);

    void updateBuffer(T* data,int count, int offset);

    int getElementCount() const {return Buffer::size/sizeof(T);}
};




template<typename T>
void TemplatedBuffer<T>::set(std::vector<T> &data, GLenum usage){
    set(data.data(),data.size(),usage);
}

template<typename T>
void TemplatedBuffer<T>::set(T* data,int count, GLenum usage){
    createGLBuffer(data,count * sizeof(T),usage);
}


template<typename T>
void TemplatedBuffer<T>::updateBuffer(T* data,int count, int offset){
    Buffer::updateBuffer(data,count*sizeof(T),offset*sizeof(T));
}
