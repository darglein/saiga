#pragma once


#include "libhello/opengl/opengl.h"
#include "libhello/opengl/buffer.h"
#include "libhello/opengl/shader.h"
#include <iostream>

#include <vector>

using std::cerr;
using std::cout;
using std::endl;

class UniformBuffer : public Buffer{
public:
    int numUniforms; //one uniform buffer can contain multiple uniforms

    UniformBuffer();
    ~UniformBuffer();


    void bind( GLuint bindingPoint);

    /**
     * Initializes this uniform buffer for the given shader.
     * If the buffer is specified with "layout (std140)" this buffer can be used
     * in multiple shaders with one init() call.
     * @param shader
     */
    void init(Shader* shader, GLuint location);

    friend std::ostream& operator<<(std::ostream& os, const UniformBuffer& ub);
};
