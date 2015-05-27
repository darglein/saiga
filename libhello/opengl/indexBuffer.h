#pragma once

#include <GL/glew.h>
#include "libhello/opengl/vertex.h"
#include <iostream>
#include <vector>




template<class index_t>
class IndexBuffer{
protected:
    int index_count;

    GLuint gl_index_buffer = 0;
public:
    IndexBuffer(){}
    ~IndexBuffer(){deleteGLBuffers();}
    void deleteGLBuffers();

    void set(std::vector<index_t> &indices);
    void set(index_t* indices,int index_count);

    void bind() const;
    void unbind() const;
};

template<class index_t>
void IndexBuffer<index_t>::deleteGLBuffers(){
    //glDeleteBuffers silently ignores 0's and names that do not correspond to existing buffer objects
    glDeleteBuffers( 1, &gl_index_buffer );
    gl_index_buffer = 0;

}


template<class index_t>
void IndexBuffer<index_t>::set(std::vector<index_t> &indices){
    set(&indices[0],indices.size());
}

template<class index_t>
void IndexBuffer<index_t>::set(index_t* indices,int index_count){

    this->index_count = index_count;

    //create IBO
    glGenBuffers( 1, &gl_index_buffer );
    glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, gl_index_buffer );
    glBufferData( GL_ELEMENT_ARRAY_BUFFER, index_count * sizeof(index_t), indices, GL_STATIC_DRAW );

    //    glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
}

template<class index_t>
void IndexBuffer<index_t>::bind() const{
    glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, gl_index_buffer );
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
