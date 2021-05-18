/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#include "saiga/core/util/DataStructures/ArrayView.h"
#include "saiga/opengl/buffer.h"
#include "saiga/opengl/opengl.h"
#include "saiga/opengl/templatedBuffer.h"

#include <memory>
namespace Saiga
{
/**
 * A Buffer Object that is used to load from and store data for a shader program is called a Shader Storage Buffer
 * Object. They can be used to share data between different shader programs, as well as quickly change between sets of
 * shader data for the same program object.
 *
 * Usage:
 *
 * 1. Create a buffer block in a GLSL shader:
 *
 * ...
 * layout (std430) buffer histogram_data
 * {
 *     float histogram[256];
 *     float median;
 * };
 * ...
 *
 *
 *
 * 2. Create shader storage buffer object
 *
 * ShaderStorageBuffer ssb;
 * ssb.createGLBuffer(&data,data_size);
 *
 *
 *
 * 3. Bind shader storage buffer to a binding point
 *
 * ssb.bind(6);
 *
 */

class SAIGA_OPENGL_API ShaderStorageBuffer : public Buffer
{
   public:
    ShaderStorageBuffer() : Buffer(GL_SHADER_STORAGE_BUFFER) {}
    ~ShaderStorageBuffer() {}

    // returns one value, the maximum size in basic machine units of a shader storage block.
    static GLint getMaxShaderStorageBlockSize()
    {
        GLint ret;
        glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &ret);
        return ret;
    }

    // returns one value, the maximum number of shader storage buffer binding points on the context.
    static GLint getMaxShaderStorageBufferBindings()
    {
        GLint ret;
        glGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS, &ret);
        return ret;
    }
};


template <class T>
class TemplatedShaderStorageBuffer : public TemplatedBuffer<T>
{
   public:
    TemplatedShaderStorageBuffer() : TemplatedBuffer<T>(GL_SHADER_STORAGE_BUFFER) {}
};

}  // namespace Saiga
