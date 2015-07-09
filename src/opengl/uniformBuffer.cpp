#include "libhello/opengl/uniformBuffer.h"


UniformBuffer::UniformBuffer() : Buffer(GL_UNIFORM_BUFFER)
{

}

UniformBuffer::~UniformBuffer()
{
}




void UniformBuffer::bind(GLuint bindingPoint)
{
    Buffer::bind();
    glBindBufferBase(target, bindingPoint, buffer);
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
