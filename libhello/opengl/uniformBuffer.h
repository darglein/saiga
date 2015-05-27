#pragma once


#include <GL/glew.h>
#include "libhello/opengl/shader.h"
#include <iostream>

#include <vector>

using std::cerr;
using std::cout;
using std::endl;

class UniformBuffer{
public:
    GLuint buffer = 0;
    GLuint size;
    int numUniforms; //one uniform buffer can contain multiple uniforms

    UniformBuffer();
    ~UniformBuffer();

    void createGLBuffer(void* data, unsigned int size );
    void deleteGLBuffer();
    void bind( GLuint bindingPoint);

    void updateBuffer(void* data, unsigned int size, unsigned int offset);


    /**
     * Initializes this uniform buffer for the given shader.
     * If the buffer is specified with "layout (std140)" this buffer can be used
     * in multiple shaders with one init() call.
     * @param shader
     */
    void init(Shader* shader, GLuint location);

    friend std::ostream& operator<<(std::ostream& os, const UniformBuffer& ub);
};
