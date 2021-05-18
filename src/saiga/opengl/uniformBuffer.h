/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#include "saiga/opengl/buffer.h"
#include "saiga/opengl/opengl.h"
#include "saiga/opengl/shader/shader.h"
#include "saiga/opengl/templatedBuffer.h"

namespace Saiga
{
/**
 * A Buffer Object that is used to store uniform data for a shader program is called a Uniform Buffer Object.
 * They can be used to share uniforms between different programs, as well as quickly change between sets of uniforms for
 * the same program object.
 *
 * Usage:
 *
 * 1. Create an uniform block in a GLSL shader:
 *
 * ...
 * layout (std140) uniform camera_data
 * {
 *     mat4 view;
 *     mat4 proj;
 *     vec4 camera_position;
 * };
 * ...
 *
 *
 *
 * 2. Query location of the uniform block
 *
 * int camera_data_location = shader.getUniformBlockLocation("camera_data");
 *
 *
 *
 * 3. Create uniform buffer object
 *
 * UniformBuffer ub;
 * ub.createGLBuffer(&data,data_size);
 *
 *
 *
 * 4. Bind uniform buffer to a binding point
 *
 * ub.bind(5);
 *
 *
 *
 * 5. Link shader uniform to the binding point
 *
 * shader.setUniformBlockBinding(camera_data_location,5);
 *
 */

class SAIGA_OPENGL_API UniformBuffer : public Buffer
{
   public:
    UniformBuffer() : Buffer(GL_UNIFORM_BUFFER) {}
    ~UniformBuffer() {}


    // returns one value, the maximum size in basic machine units of a uniform block, which must be at least 16384.
    static GLint getMaxUniformBlockSize()
    {
        GLint ret;
        glGetIntegerv(GL_MAX_UNIFORM_BLOCK_SIZE, &ret);
        return ret;
    }
    // returns one value, the maximum number of uniform buffer binding points on the context, which must be at least 36.
    static GLint getMaxUniformBufferBindings()
    {
        GLint ret;
        glGetIntegerv(GL_MAX_UNIFORM_BUFFER_BINDINGS, &ret);
        return ret;
    }
};


template <class T>
class TemplatedUniformBuffer : public TemplatedBuffer<T>
{
   public:
    TemplatedUniformBuffer() : TemplatedBuffer<T>(GL_UNIFORM_BUFFER) {}
};


}  // namespace Saiga
