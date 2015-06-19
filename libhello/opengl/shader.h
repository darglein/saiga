
#pragma once

#include "libhello/opengl/opengl.h"


#include "libhello/util/glm.h"

using std::cerr;
using std::string;

/**
 * Currently supported shader types: GL_VERTEX_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER
 * @brief The Shader class
 */


class Shader{
public:
    /**
     * @brief The CodeInjection class
     * The code injections are added to the top of the shader.
     * This can for exapmple be used to add different #define.
     */
    class CodeInjection{
    public:
        //shader type, must be one of the supported types
        int type;
        std::string code;
    };
    string name;
    string shaderPath;
    string prefix;
    string typeToName(GLenum type);

    Shader();
    virtual ~Shader();
    Shader(const string &multi_file);

    //Shader loading utility programs
    void printProgramLog( GLuint program );
    void printShaderLog( GLuint shader );


    std::vector<string> loadAndPreproccess(const string &file);
    bool addMultiShaderFromFile(const string &multi_file);
    GLuint addShader(const char* content, GLenum type);
    GLuint addShaderFromFile(const char* file, GLenum type);
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




