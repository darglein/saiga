
#pragma once

#include "saiga/opengl/opengl.h"
#include "saiga/util/glm.h"
#include "saiga/opengl/shaderpart.h"

#include <vector>
#include <tuple>

using std::cerr;
using std::string;

/**
 * Currently supported shader types: GL_VERTEX_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER
 * @brief The Shader class
 */


class SAIGA_GLOBAL Shader{
public:


//    typedef std::vector<std::tuple<GLenum,std::string,int>> ShaderCodeInjections;
        typedef std::vector<ShaderCodeInjection> ShaderCodeInjections;

    std::string name;
    std::string shaderPath;
    std::string prefix;
//    ShaderCodeInjections injections;
    ShaderCodeInjections injections;

    std::vector<std::string> vertexShaderCode;
    std::vector<std::string> geometryShaderCode;
    std::vector<std::string> fragmentShaderCode;

    Shader();
    virtual ~Shader();
    Shader(const std::string &multi_file);

    void addShaderCodeInjection(const ShaderCodeInjection& sci){injections.push_back(sci);}
    void setShaderCodeInjections(const ShaderCodeInjections& sci){injections=sci;}
    void addInjectionsToCode(GLenum type, std::vector<std::string> &content);

    std::string typeToName(GLenum type);
    //Shader loading utility programs
    void printProgramLog( GLuint program );
    void printShaderLog(GLuint shader , GLenum type);
    void parseShaderError(const std::string& message, GLenum type);


    std::vector<string> loadAndPreproccess(const std::string &file);
    bool addMultiShaderFromFile(const std::string &multi_file);
    GLuint addShader(std::vector<std::string> &content, GLenum type);
    GLuint addShaderFromFile(const std::string& file, GLenum type);
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

    GLuint program, vertShader,geoShader,fragShader;
    static Shader* getShader(int id); //time o(1)
    static void reloadAll();
    static void clearShaders(); //deleted all shaders
    static void createShaders(); //creates all shaders


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




