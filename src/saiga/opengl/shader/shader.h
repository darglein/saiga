/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/math.h"
#include "saiga/opengl/opengl.h"
#include "saiga/opengl/shader/UniformHelper.h"
#include "saiga/opengl/shader/shaderpart.h"
#include "saiga/opengl/texture/Texture2D.h"

#include <filesystem>
#include <memory>
#include <vector>

namespace Saiga
{
class TextureBase;


/**
 * A shader object represents a 'program' in OpenGL terminology.
 */

class SAIGA_OPENGL_API Shader
{
   private:
    /**
     * The program id of the currently bound shader. In debug and testing mode all shader functions,
     * that require the shader to be bound check this id with the own id.
     *
     * Additionally unbind has to be called for every bind. A new shader bind, when an
     * other shader is already bound will result in an error in debug and testing mode.
     */
    static GLuint boundShader;

   public:
    static GLuint getBoundShader() { return boundShader; }
    static bool add_glsl_line_directives;

   public:
    GLuint program = 0;
    std::vector<std::shared_ptr<ShaderPart>> shaders;
    std::string file;
    ShaderCodeInjections injections;

    // all files that are referenced by includes
    // important for automatic shader reloading
    std::vector<std::pair<std::string, std::filesystem::file_time_type>> dependent_files_and_date;


    Shader();
    virtual ~Shader();
    Shader(Shader const&) = delete;
    Shader& operator=(Shader const&) = delete;

    bool init(const std::string& file, const ShaderCodeInjections& injections = {});
    bool reload();

    // ===================================== program stuff =====================================

    /**
     * Creates the underlying gl-program, attaches all shader parts to it and links the shader.
     * The ShaderParts have to be compiled without errors before this step.
     * Returns the program id or 0 if the link fails.
     *
     * Calls checkUniforms after succesfully linking.
     */
    GLuint createProgram();

    /**
     * Deletes the underlying gl-program. The shaderparts are not deleted, so it could be linked again.
     */
    void destroyProgram();

    /**
     * Checks the programs info logs and returns true when no error has occured.
     */
    bool printProgramLog();

    /**
     * Binds this shader to opengl. Maps to glUseProgram
     */
    [[nodiscard]] bool bind();

    /**
     * Binds the default program (0) to opengl.
     * Does nothing in testing and release mode.
     */
    void unbind();

    /**
     * Returns true if this shader is the last shader that has called bind().
     */
    bool isBound();


    /**
     * Returns the binary code of this program and the internal format.
     */
    bool getBinary(std::vector<uint8_t>& binary, GLenum& format);

    bool setBinary(const std::vector<uint8_t>& binary, GLenum format);

    // ===================================== Compute Shaders =====================================
    // Note: Compute shaders require OpenGL 4.3+

    /**
     * Returns the work group size specified in the shader.
     *
     * For example:
     * layout(local_size_x = 32, local_size_y = 32) in;
     *
     * returns uvec3(32,32,1)
     */

    uvec3 getComputeWorkGroupSize();

    /**
     * Calculates the number of groups required for the given problem size.
     * The number of shader executions will then be greater or equal to the problem size.
     */

    uvec3 getNumGroupsCeil(const uvec3& problem_size);
    uvec3 getNumGroupsCeil(const uvec3& problem_size, const uvec3& work_group_size);

    /**
     * Initates the compute operation.
     * The shader must be bound beforehand.
     */
    void dispatchCompute(const uvec3& num_groups);
    void dispatchCompute(GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z);

    // Dispatches as many blocks as needed to cover the texture with 1 thread per pixel
    void dispatchComputeImage(Texture2D* texture, int local_size);

    /**
     * Directly maps to glMemoryBarrier.
     * See here https://www.opengl.org/sdk/docs/man/html/glMemoryBarrier[0]html
     *
     * The 2 most common barries have their own function, see below.
     */

    static void memoryBarrier(MemoryBarrierMask barriers);

    /**
     * Calls memoryBarrier with GL_TEXTURE_FETCH_BARRIER_BIT.
     *
     * Texture fetches from shaders, including fetches from buffer object memory via buffer textures,
     * after the barrier will reflect data written by shaders prior to the barrier.
     */

    static void memoryBarrierTextureFetch();

    /**
     * Calls memoryBarrier with GL_SHADER_IMAGE_ACCESS_BARRIER_BIT.
     *
     * Memory accesses using shader image load, store, and atomic built-in functions issued after the barrier
     * will reflect data written by shaders prior to the barrier. Additionally, image stores and atomics issued
     * after the barrier will not execute until all memory accesses (e.g., loads, stores, texture fetches, vertex
     * fetches) initiated prior to the barrier complete.
     */

    static void memoryBarrierImageAccess();

    // ===================================== uniforms =====================================

    GLint getUniformLocation(const char* name);
    void getUniformInfo(GLuint location);
    virtual void checkUniforms() {}



    // ===================================== uniform blocks =====================================

    GLuint getUniformBlockLocation(const char* name);
    void setUniformBlockBinding(GLuint blockLocation, GLuint bindingPoint);
    // size of the complete block in bytes
    GLint getUniformBlockSize(GLuint blockLocation);
    std::vector<GLint> getUniformBlockIndices(GLuint blockLocation);
    std::vector<GLint> getUniformBlockSize(std::vector<GLint> indices);
    std::vector<GLint> getUniformBlockType(std::vector<GLint> indices);
    std::vector<GLint> getUniformBlockOffset(std::vector<GLint> indices);


    // ===================================== uniform uploads =====================================

    /**     * Uploading a uniform to video ram.
     * Maps to glUniform**.
     * The shader must be bound beforehand.
     */

    template <class T>
    void upload(int location, const T& value)
    {
        SAIGA_ASSERT(isBound());
        UniformHelper<T>::Set(location, 1, &value);
    }

    template <class T>
    void upload(int location, int count, const T* values)
    {
        SAIGA_ASSERT(isBound());
        UniformHelper<T>::Set(location, count, values);
    }

    // binds the texture to the given texture unit and sets the uniform.
    void upload(int location, TextureBase* texture, int textureUnit) { upload(location, *texture, textureUnit); }
    void upload(int location, std::shared_ptr<TextureBase> texture, int textureUnit)
    {
        upload(location, *texture, textureUnit);
    }
    void upload(int location, TextureBase& texture, int textureUnit)
    {
        SAIGA_ASSERT(isBound());
        UniformHelper<TextureBase>::Set(location, textureUnit, &texture);
    }
};

}  // namespace Saiga
