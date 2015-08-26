#pragma once

#include "saiga/opengl/opengl.h"
#include "saiga/util/glm.h"
#include "saiga/opengl/shader/shaderpart.h"

#include <vector>
#include <memory>

class raw_Texture;




class SAIGA_GLOBAL Shader{
public:

    GLuint program = 0;
    std::vector<std::shared_ptr<ShaderPart>> shaders;


    Shader();
    virtual ~Shader();


    // ================== program stuff ==================

    void bind();
    void unbind();
    GLuint createProgram();
    void printProgramLog();


    // ================== uniforms ==================

    GLint getUniformLocation(const char* name);
    void getUniformInfo(GLuint location);
    virtual void checkUniforms(){}



    // ================== uniform blocks ==================

    GLuint getUniformBlockLocation(const char* name);
    void setUniformBlockBinding(GLuint blockLocation, GLuint bindingPoint);
    //size of the complete block in bytes
    GLint getUniformBlockSize(GLuint blockLocation);
    std::vector<GLint> getUniformBlockIndices(GLuint blockLocation);
    std::vector<GLint> getUniformBlockSize(GLuint blockLocation, std::vector<GLint> indices);
    std::vector<GLint> getUniformBlockType(GLuint blockLocation, std::vector<GLint> indices);
    std::vector<GLint> getUniformBlockOffset(GLuint blockLocation, std::vector<GLint> indices);


    // ================== uniform uploads ==================

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




