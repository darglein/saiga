#pragma once

#include "saiga/util/glm.h"
#include "saiga/opengl/buffer.h"


#include <vector>


template<typename data_t>
class InstancedBuffer : protected Buffer{
public:
    int elements;

    InstancedBuffer():Buffer(GL_ARRAY_BUFFER){}
    ~InstancedBuffer(){}

    void createGLBuffer(unsigned int elements=0);
    void updateBuffer(void* data, unsigned int elements, unsigned int offset);

    void setAttributes(int location, int divisor=1);
};


template<typename data_t>
void InstancedBuffer<data_t>::createGLBuffer(unsigned int elements)
{
    Buffer::createGLBuffer(nullptr,elements*sizeof(data_t),GL_DYNAMIC_DRAW);
    this->elements = elements;
}

template<typename data_t>
void InstancedBuffer<data_t>::updateBuffer(void *data, unsigned int elements, unsigned int offset)
{
    Buffer::updateBuffer(data,elements*sizeof(data_t),offset*sizeof(data_t));
}



template<>
inline void InstancedBuffer<glm::mat4>::setAttributes(int location, int divisor)
{
    Buffer::bind();

    for (unsigned int i = 0; i < 4 ; i++) {
        glEnableVertexAttribArray(location + i);
        glVertexAttribPointer(location + i, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4),
                              (const GLvoid*)(sizeof(GLfloat) * i * 4));
        glVertexAttribDivisor(location + i, divisor);
    }
}
