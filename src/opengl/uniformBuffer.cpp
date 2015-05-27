#include "opengl/uniformBuffer.h"


UniformBuffer::UniformBuffer()
{

}

UniformBuffer::~UniformBuffer()
{
    deleteGLBuffer();
}

void UniformBuffer::createGLBuffer(void *data, unsigned int size)
{
    glGenBuffers( 1, &buffer );

    glBindBuffer( GL_UNIFORM_BUFFER, buffer );

    glBufferData(GL_UNIFORM_BUFFER, size, data, GL_DYNAMIC_DRAW);

}

void UniformBuffer::deleteGLBuffer()
{
    if(buffer){
        glDeleteBuffers( 1, &buffer );
        buffer = 0;
    }
}



void UniformBuffer::bind(GLuint bindingPoint)
{
    glBindBuffer( GL_UNIFORM_BUFFER, buffer );
    glBindBufferBase(GL_UNIFORM_BUFFER, bindingPoint, buffer);
}

void UniformBuffer::updateBuffer(void *data, unsigned int size, unsigned int offset)
{
    glBindBuffer( GL_UNIFORM_BUFFER, buffer );
    glBufferSubData(GL_UNIFORM_BUFFER,offset,size,data);
}

void UniformBuffer::init(Shader *shader, GLuint location)
{
    size = shader->getUniformBlockSize(location);

    std::vector<GLint> indices = shader->getUniformBlockIndices(location);
    numUniforms = indices.size();

    createGLBuffer(nullptr,size);

}

std::ostream &operator<<(std::ostream &os, const UniformBuffer &ub){
    os<<"UniformBuffer "<<"size="<<ub.size<<" numUniforms="<<ub.numUniforms;
    return os;
}
