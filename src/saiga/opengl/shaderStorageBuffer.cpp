/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/shaderStorageBuffer.h"
#include <iostream>

namespace Saiga
{
ShaderStorageBuffer::ShaderStorageBuffer() : Buffer(GL_SHADER_STORAGE_BUFFER) {}

ShaderStorageBuffer::~ShaderStorageBuffer() {}

std::ostream& operator<<(std::ostream& os, const ShaderStorageBuffer& ssb)
{
    os << "ShaderStorageBuffer "
       << "size=dynamic";
    return os;
}

GLint ShaderStorageBuffer::getMaxShaderStorageBlockSize()
{
    GLint ret;
    glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &ret);
    return ret;
}

GLint ShaderStorageBuffer::getMaxShaderStorageBufferBindings()
{
    GLint ret;
    glGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS, &ret);
    return ret;
}

}  // namespace Saiga
