#pragma once

#include "saiga/opengl/opengl.h"
#include "saiga/util/glm.h"
#include "saiga/opengl/shader/shaderpart.h"

#include <vector>
#include <memory>

class raw_Texture;


/**
 * A shader object represents a 'program' in OpenGL terminology.
 */

class SAIGA_GLOBAL Shader{
public:

    GLuint program = 0;
    std::vector<std::shared_ptr<ShaderPart>> shaders;


    Shader();
    virtual ~Shader();


    // ===================================== program stuff =====================================

    void bind();
    void unbind();
    GLuint createProgram();
    void destroyProgram();
    void printProgramLog();

    // ===================================== Compute Shaders =====================================
    // Note: Compute shaders require OpenGL 4.3+

    /**
     * Returns the work group size specified in the shader.
     *
     * For example:
     * layout(local_size_x = 32, local_size_y = 32) in;
     *
     * returns glm::uvec3(32,32,1)
     */

    glm::uvec3 getComputeWorkGroupSize();

    /**
     * Calculates the number of groups required for the given problem size.
     * The number of shader executions will then be greater or equal to the problem size.
     */

    glm::uvec3 getNumGroupsCeil( const glm::uvec3 &problem_size);
    glm::uvec3 getNumGroupsCeil( const glm::uvec3 &problem_size,const glm::uvec3 &work_group_size);

    /**
     * Initates the compute operation.
     * The shader must be bound beforehand.
     */

    void dispatchCompute(const glm::uvec3 &num_groups);
    void dispatchCompute(GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z);

    /**
     * Directly maps to glMemoryBarrier.
     * See here https://www.opengl.org/sdk/docs/man/html/glMemoryBarrier.xhtml
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
     * after the barrier will not execute until all memory accesses (e.g., loads, stores, texture fetches, vertex fetches)
     * initiated prior to the barrier complete.
     */

    static void memoryBarrierImageAccess();

    // ===================================== uniforms =====================================

    GLint getUniformLocation(const char* name);
    void getUniformInfo(GLuint location);
    virtual void checkUniforms(){}



    // ===================================== uniform blocks =====================================

    GLuint getUniformBlockLocation(const char* name);
    void setUniformBlockBinding(GLuint blockLocation, GLuint bindingPoint);
    //size of the complete block in bytes
    GLint getUniformBlockSize(GLuint blockLocation);
    std::vector<GLint> getUniformBlockIndices(GLuint blockLocation);
    std::vector<GLint> getUniformBlockSize(GLuint blockLocation, std::vector<GLint> indices);
    std::vector<GLint> getUniformBlockType(GLuint blockLocation, std::vector<GLint> indices);
    std::vector<GLint> getUniformBlockOffset(GLuint blockLocation, std::vector<GLint> indices);


    // ===================================== uniform uploads =====================================

    void upload(int location, const mat4 &m);
    void upload(int location, const vec4 &v);
    void upload(int location, const vec3 &v);
    void upload(int location, const vec2 &v);
    void upload(int location, const int &v);
    void upload(int location, const float &f);
    //array uploads
    void upload(int location, int count, mat4* m);
    void upload(int location, int count, vec4* v);
    void upload(int location, int count, vec3* v);
    void upload(int location, int count, vec2* v);

    //binds the texture to the given texture unit and sets the uniform.
    void upload(int location, raw_Texture *texture, int textureUnit);
};




