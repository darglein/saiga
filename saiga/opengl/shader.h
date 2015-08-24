#pragma once

#include "saiga/opengl/opengl.h"
#include "saiga/util/glm.h"
#include "saiga/opengl/shaderpart.h"

#include <vector>


/**
 * Currently supported shader types: GL_VERTEX_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER
 * @brief The Shader class
 */


class SAIGA_GLOBAL Shader{
public:


    typedef std::vector<ShaderCodeInjection> ShaderCodeInjections;

    std::string name;
    std::string shaderPath;
    std::string prefix;

    ShaderCodeInjections injections;


    GLuint program = 0;

    std::vector<ShaderPart> shaders;


    Shader();
    virtual ~Shader();
    Shader(const std::string &multi_file);

    void addShaderCodeInjection(const ShaderCodeInjection& sci){injections.push_back(sci);}
    void setShaderCodeInjections(const ShaderCodeInjections& sci){injections=sci;}


    void printProgramLog( GLuint program );


    std::vector<std::string> loadAndPreproccess(const std::string &file);
    bool addMultiShaderFromFile(const std::string &multi_file);
    void addShader(std::vector<std::string> &content, GLenum type);
    void addShaderFromFile(const std::string& file, GLenum type);
    GLuint createProgram();
    GLint getUniformLocation(const char* name);

    void getUniformInfo(GLuint location);
    //uniform blocks
    GLuint getUniformBlockLocation(const char* name);
    void setUniformBlockBinding(GLuint blockLocation, GLuint bindingPoint);

    //size of the complete block in bytes
    GLint getUniformBlockSize(GLuint blockLocation);

    std::vector<GLint> getUniformBlockIndices(GLuint blockLocation);
    std::vector<GLint> getUniformBlockSize(GLuint blockLocation, std::vector<GLint> indices);
    std::vector<GLint> getUniformBlockType(GLuint blockLocation, std::vector<GLint> indices);
    std::vector<GLint> getUniformBlockOffset(GLuint blockLocation, std::vector<GLint> indices);

public:



    bool reload();
    void bind();
    void unbind();


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

    virtual void checkUniforms(){}
};




