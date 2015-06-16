#pragma once

#include "libhello/opengl/opengl.h"


#include "libhello/util/glm.h"
#include "libhello/util/loader.h"

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
    string typeToName(int type);

    Shader();
    virtual ~Shader();
    Shader(const string &multi_file);

    //Shader loading utility programs
    void printProgramLog( GLuint program );
    void printShaderLog( GLuint shader );


    std::vector<string> loadAndPreproccess(const string &file);
    bool addMultiShaderFromFile(const string &multi_file);
    GLuint addShader(const char* content, int type);
    GLuint addShaderFromFile(const char* file, int type);
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

class ShaderLoader : public Loader<Shader>{
public:
    Shader* loadFromFile(const std::string &name);
    template<typename shader_t> shader_t* load(const std::string &name);
    template<typename shader_t> shader_t* loadFromFile(const std::string &name, const std::string &prefix);
    void reload();
};




template<typename shader_t>
shader_t* ShaderLoader::load(const std::string &name){
    shader_t* object;
    //check if already exists
    for(Shader* &obj : objects){
        if(obj->name == name){
            object = dynamic_cast<shader_t*>(obj);
            if(object != nullptr){
                return object;
            }
        }
    }


    for(std::string &path : locations){
        std::string complete_path = path + "/" + name;
        object = loadFromFile<shader_t>(complete_path,path);
        if (object){
            object->name = name;
            std::cout<<"Loaded from file: "<<complete_path<<std::endl;
            objects.push_back(object);
            return object;
        }
    }

    std::cout<<"Failed to load "<<name<<"!!!"<<std::endl;
    exit(0);
    return NULL;
}

template<typename shader_t>
shader_t* ShaderLoader::loadFromFile(const std::string &name, const std::string &prefix){
    shader_t* shader = new shader_t(name);
    shader->prefix = prefix;
    if(shader->reload()){
        return shader;
    }
    delete shader;
    return NULL;
}


