#pragma once

#include "libhello/opengl/buffer.h"

#include <GL/glew.h>
#include "libhello/opengl/vertex.h"
#include <iostream>
#include <vector>




template<class index_t>
class IndexBuffer : public Buffer{
public:
    int index_count;

public:
    IndexBuffer(): Buffer(GL_ELEMENT_ARRAY_BUFFER){}
    ~IndexBuffer(){}

    void set(std::vector<index_t> &indices);
    void set(index_t* indices,int index_count);

    void unbind() const;
};




template<class index_t>
void IndexBuffer<index_t>::set(std::vector<index_t> &indices){
    set(&indices[0],indices.size());
}

template<class index_t>
void IndexBuffer<index_t>::set(index_t* indices,int index_count){

    this->index_count = index_count;

    createGLBuffer(indices,index_count * sizeof(index_t),GL_STATIC_DRAW);

}

template<class index_t>
void IndexBuffer<index_t>::unbind() const{
    //Binding 0 is deprecated!

    // Application-generated object names - the names of all object types, such as buffer, query, and texture objects,
    // must be generated using the corresponding Gen* commands. Trying to bind an object name not returned by a Gen*command will
    // result in an INVALID_OPERATION error. This behavior is already the case for framebuffer, renderbuffer, and vertex array objects.
    // Object types which have default objects (objects named zero), such as vertex array, framebuffer, and texture objects,
    // may also bind the default object, even though it is not returned by Gen*.

    //      glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
}
