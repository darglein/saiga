/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/shader/shader.h"

#include "saiga/core/rendering/ShaderParser.h"
#include "saiga/core/util/assert.h"
#include "saiga/opengl/error.h"
#include "saiga/opengl/texture/TextureBase.h"

#include <algorithm>
#include <fstream>

namespace Saiga
{
GLuint Shader::boundShader            = 0;
bool Shader::add_glsl_line_directives = false;

Shader::Shader() {}

Shader::~Shader()
{
    destroyProgram();
}

bool Shader::reload()
{
    bool need_reload = false;
    for (auto& f_t : dependent_files_and_date)
    {
        SAIGA_ASSERT(std::filesystem::exists(f_t.first));
        auto date = std::filesystem::last_write_time(f_t.first);

        if (date > f_t.second)
        {
            need_reload = true;
        }
    }

    if (need_reload)
    {
        return init(file, injections);
    }
    else
    {
        return true;
    }
}

bool Shader::init(const std::string& file, const ShaderCodeInjections& injections)
{
    std::cout << "loading shader " << file << std::endl;
    this->file       = file;
    this->injections = injections;
    shaders.clear();
    dependent_files_and_date.clear();

    auto c = LoadFileAndResolveIncludes(file, add_glsl_line_directives);
    if (!c.valid) return false;

    for (auto f : c.dependent_files)
    {
        SAIGA_ASSERT(std::filesystem::exists(f));
        auto date = std::filesystem::last_write_time(f);
        dependent_files_and_date.emplace_back(f, date);
    }

    for (auto p : c.parts)
    {
        if (p.type.empty() || p.end - p.start == 0) continue;
        GLenum gl_type = GL_NONE;
        for (int i = 0; i < ShaderPart::shaderTypeCount; ++i)
        {
            if (p.type == ShaderPart::shaderTypeStrings[i])
            {
                gl_type = ShaderPart::shaderTypes[i];
            }
        }

        if (gl_type == GL_NONE)
        {
            SAIGA_EXIT_ERROR("Unknown shader type: " + p.type);
        }

        std::vector<std::string> content(c.code.begin() + p.start, c.code.begin() + p.end);

        auto shader = std::make_shared<ShaderPart>(content, gl_type, injections);
        shaders.push_back(shader);
    }

    createProgram();
    return true;
}

// ===================================== program stuff =====================================


GLuint Shader::createProgram()
{
    for (auto& sp : shaders)
    {
        if (!sp->valid)
        {
            destroyProgram();
            return 0;
        }
    }

    program = glCreateProgram();

    // attach all shaders
    for (auto& sp : shaders)
    {
        glAttachShader(program, sp->id);
    }
    assert_no_glerror();
    glLinkProgram(program);
    assert_no_glerror();
    printProgramLog();


    assert_no_glerror();
    for (auto& sp : shaders)
    {
        sp->deleteGLShader();
    }

    // Check if linking was actually successful
    GLint isLinked = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &isLinked);
    if (isLinked)
    {
        checkUniforms();
        return program;
    }
    else
    {
        destroyProgram();
        return 0;
    }
}

void Shader::destroyProgram()
{
    if (program != 0)
    {
        glDeleteProgram(program);
        program = 0;
    }
}

bool Shader::printProgramLog()
{
    // Make sure name is shader
    if (glIsProgram(program) == GL_TRUE)
    {
        // Program log length
        int infoLogLength = 0;
        int maxLength     = infoLogLength;

        // Get info std::string length
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &maxLength);

        // Allocate std::string
        char* infoLog = new char[maxLength];

        // Get info log
        glGetProgramInfoLog(program, maxLength, &infoLogLength, infoLog);
        if (infoLogLength > 0)
        {
            // Print Log
            std::cout << "Program info log:" << std::endl;
            std::cout << infoLog << std::endl;
        }
        assert_no_glerror();

        // Deallocate std::string
        delete[] infoLog;
        return infoLogLength == 0;
    }
    else
    {
        std::cout << "Name " << program << " is not a program" << std::endl;
        return false;
    }
}

bool Shader::bind()
{
    if (program)
    {
        // allow double bind
        SAIGA_ASSERT(boundShader == program || boundShader == 0);
        boundShader = program;
        glUseProgram(program);
        assert_no_glerror();
        return true;
    }
    else
    {
        return false;
    }
}

void Shader::unbind()
{
    // allow double unbind
    SAIGA_ASSERT(boundShader == program || boundShader == 0);
    boundShader = 0;
#if defined(SAIGA_DEBUG)
    glUseProgram(0);
#endif
    assert_no_glerror();
}

bool Shader::isBound()
{
    return boundShader == program;
}

bool Shader::getBinary(std::vector<uint8_t>& binary, GLenum& format)
{
#if 0
    GLint num_program_binary_formats[1];
     glGetIntegerv(GL_NUM_PROGRAM_BINARY_FORMATS,num_program_binary_formats);

     std::vector<GLint> program_binary_formats(num_program_binary_formats[0]);
     glGetIntegerv(GL_PROGRAM_BINARY_FORMATS,program_binary_formats.data());

     std::cout << "Num binary formats: " << num_program_binary_formats[0] << std::endl;
     for(auto f : program_binary_formats){
         std::cout << f << std::endl;
     }
#endif

    GLint length[1];
    glGetProgramiv(program, GL_PROGRAM_BINARY_LENGTH, length);
    assert_no_glerror();
    GLsizei size = length[0];
    if (size == 0)
    {
        // When a progam's GL_LINK_STATUS is GL_FALSE, its program binary length is zero.
        return false;
    }

    binary.resize(size);
    //    std::cout << "Binary length: " << size << std::endl;
    GLsizei actualLength;
    glGetProgramBinary(program, size, &actualLength, &format, binary.data());

    SAIGA_ASSERT(size == actualLength);
    //    std::cout << "recieved format: " << format << std::endl;
    //    std::cout << "actualLength: " << actualLength << std::endl;
    assert_no_glerror();
    return true;
}

bool Shader::setBinary(const std::vector<uint8_t>& binary, GLenum format)
{
    glProgramBinary(program, format, binary.data(), binary.size());
    assert_no_glerror();
    return true;
}

// ===================================== Compute Shaders =====================================


uvec3 Shader::getComputeWorkGroupSize()
{
    GLint work_size[3];
    glGetProgramiv(program, GL_COMPUTE_WORK_GROUP_SIZE, work_size);
    assert_no_glerror();
    return uvec3(work_size[0], work_size[1], work_size[2]);
}

uvec3 Shader::getNumGroupsCeil(const uvec3& problem_size)
{
    return getNumGroupsCeil(problem_size, getComputeWorkGroupSize());
}

uvec3 Shader::getNumGroupsCeil(const uvec3& problem_size, const uvec3& work_group_size)
{
    //    uvec3 ret = problem_size/work_group_size;
    //    uvec3 rest = problem_size%work_group_size;

    //    ret[0] += rest[0] ? 1 : 0;
    //    ret[1] += rest[1] ? 1 : 0;
    //    ret[2] += rest[2] ? 1 : 0;

    //    return ret;

    SAIGA_EXIT_ERROR("sldg");
    return problem_size;  //(problem_size + work_group_size - uvec3(1)) / (work_group_size);
}

void Shader::dispatchCompute(const uvec3& num_groups)
{
    SAIGA_ASSERT(isBound());
    dispatchCompute(num_groups[0], num_groups[1], num_groups[2]);
    assert_no_glerror();
}

void Shader::dispatchCompute(GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z)
{
    SAIGA_ASSERT(isBound());
    glDispatchCompute(num_groups_x, num_groups_y, num_groups_z);
    assert_no_glerror();
}

void Shader::dispatchComputeImage(Texture2D* texture, int local_size)
{
    int gw = iDivUp(texture->getWidth(), local_size);
    int gh = iDivUp(texture->getHeight(), local_size);
    dispatchCompute(uvec3(gw, gh, 1));
}


void Shader::memoryBarrier(MemoryBarrierMask barriers)
{
    glMemoryBarrier(barriers);
    assert_no_glerror();
}

void Shader::memoryBarrierTextureFetch()
{
    memoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
    assert_no_glerror();
}

void Shader::memoryBarrierImageAccess()
{
    memoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    assert_no_glerror();
}

// ===================================== uniforms =====================================


GLint Shader::getUniformLocation(const char* name)
{
    GLint i = glGetUniformLocation(program, name);
    return i;
}

void Shader::getUniformInfo(GLuint location)
{
    const GLsizei bufSize = 128;

    GLsizei length;
    GLint size;
    GLenum type;
    GLchar name[bufSize];

    glGetActiveUniform(program, location, bufSize, &length, &size, &type, name);

    std::cout << "uniform info " << location << std::endl;
    std::cout << "name " << name << std::endl;
    //    std::cout<<"length "<<length<<endl;
    std::cout << "size " << size << std::endl;
    std::cout << "type " << (int)type << std::endl;
}

// ===================================== uniform blocks =====================================

GLuint Shader::getUniformBlockLocation(const char* name)
{
    GLuint blockIndex = glGetUniformBlockIndex(program, name);

    if (blockIndex == GL_INVALID_INDEX)
    {
        std::cout << "Warning: glGetUniformBlockIndex: uniform block '" << name << "' invalid!" << std::endl;
    }
    assert_no_glerror();
    return blockIndex;
}

void Shader::setUniformBlockBinding(GLuint blockLocation, GLuint bindingPoint)
{
    glUniformBlockBinding(program, blockLocation, bindingPoint);
    assert_no_glerror();
}

GLint Shader::getUniformBlockSize(GLuint blockLocation)
{
    GLint ret;
    glGetActiveUniformBlockiv(program, blockLocation, GL_UNIFORM_BLOCK_DATA_SIZE, &ret);
    return ret;
}

std::vector<GLint> Shader::getUniformBlockIndices(GLuint blockLocation)
{
    GLint ret;
    glGetActiveUniformBlockiv(program, blockLocation, GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS, &ret);

    std::vector<GLint> indices(ret);
    glGetActiveUniformBlockiv(program, blockLocation, GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES, &indices[0]);

    return indices;
}

std::vector<GLint> Shader::getUniformBlockSize(std::vector<GLint> indices)
{
    std::vector<GLint> ret(indices.size());
    glGetActiveUniformsiv(program, indices.size(), (GLuint*)indices.data(), GL_UNIFORM_SIZE, ret.data());
    return ret;
}

std::vector<GLint> Shader::getUniformBlockType(std::vector<GLint> indices)
{
    std::vector<GLint> ret(indices.size());
    glGetActiveUniformsiv(program, indices.size(), (GLuint*)indices.data(), GL_UNIFORM_TYPE, ret.data());
    return ret;
}

std::vector<GLint> Shader::getUniformBlockOffset(std::vector<GLint> indices)
{
    std::vector<GLint> ret(indices.size());
    glGetActiveUniformsiv(program, indices.size(), (GLuint*)indices.data(), GL_UNIFORM_OFFSET, ret.data());
    return ret;
}


}  // namespace Saiga
