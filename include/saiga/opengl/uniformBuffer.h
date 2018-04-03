/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#include "saiga/opengl/opengl.h"
#include "saiga/opengl/buffer.h"
#include "saiga/opengl/shader/shader.h"

namespace Saiga {

/**
 * A Buffer Object that is used to store uniform data for a shader program is called a Uniform Buffer Object.
 * They can be used to share uniforms between different programs, as well as quickly change between sets of uniforms for the same program object.
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

class SAIGA_GLOBAL UniformBuffer : public Buffer{
public:
    UniformBuffer();
    ~UniformBuffer();

    /**
     * This function is obsolete and may be removed in later versions.
     *
     * Initializes this uniform buffer for the given shader.
     * If the buffer is specified with "layout (std140)" this buffer can be used
     * in multiple shaders with one init() call.
     * @param shader
     */
    void init(std::shared_ptr<Shader>  shader, GLuint location);


    friend std::ostream& operator<<(std::ostream& os, const UniformBuffer& ub);

    //returns one value, the maximum size in basic machine units of a uniform block, which must be at least 16384.
    static GLint getMaxUniformBlockSize();
    //returns one value, the maximum number of uniform buffer binding points on the context, which must be at least 36.
    static GLint getMaxUniformBufferBindings();

};

}
