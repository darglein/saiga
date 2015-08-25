#include "saiga/opengl/shader/shader.h"

#include "saiga/util/error.h"
#include <fstream>
#include <algorithm>

#define STATUS_WAITING 0 //waiting for "start"
#define STATUS_START 1 //found start + looking for type
#define STATUS_READING 2 //found start + found type
#define STATUS_ERROR 3

Shader::Shader() : Shader(""){
}

Shader::~Shader(){
    if(program!=0){
        glDeleteProgram(program);
        program = 0;
    }

    for(ShaderPart& sp : shaders){
        sp.deleteGLShader();
    }

}

Shader::Shader(const std::string &multi_file) : shaderPath(multi_file){

}




bool Shader::reload(){
    if(shaderPath.length()<=0){
        std::cerr<<"Reload only works if the Shader object is created with a multishader in the constructor"<<endl;
        return false;
    }
    if(program!=0){
        glDeleteProgram(program);
        program = 0;
    }

    return addMultiShaderFromFile(shaderPath);


}

std::vector<std::string> Shader::loadAndPreproccess(const std::string &file)
{
    std::vector<std::string> ret;

    std::ifstream fileStream(file, std::ios::in);
    if(!fileStream.is_open()) {
        return ret;
    }

    const std::string include("#include ");

    while(!fileStream.eof()) {
        std::string line;
        std::getline(fileStream, line);

        if(include.size()<line.size() && line.compare(0, include.length(), include)==0){
            line = line.substr(include.size()-1);

            auto it = std::remove(line.begin(),line.end(),'"');
            line.erase(it,line.end());

            it = std::remove(line.begin(),line.end(),' ');
            line.erase(it,line.end());

            //recursivly load includes
            std::vector<std::string> tmp = loadAndPreproccess(prefix+"/"+line);
            ret.insert(ret.end(),tmp.begin(),tmp.end());
            //            cout<<"shader path "<<shaderPath<<endl;
        }else{
            ret.push_back(line);
        }


    }
    return ret;
}

bool Shader::addMultiShaderFromFile(const std::string &multi_file) {

    std::string content,errorMsg;


    std::vector<std::string> data = loadAndPreproccess(multi_file);

    std::vector<std::string> code;
    //    cout<<"Preproccess finished. "<<data.size()<<" lines"<<endl;

    if(data.size()<=0)
        return false;

    int status = STATUS_WAITING;
    GLenum type = GL_INVALID_ENUM;
    int lineCount =0;

    for(std::string line : data){
        lineCount++;
        if(line.compare("##start")==0){
            status = (status==STATUS_WAITING)?STATUS_START:STATUS_ERROR;

        }else if(line.compare("##end")==0){
            status = (status==STATUS_READING)?STATUS_WAITING:STATUS_ERROR;

            if(status != STATUS_ERROR){
                //reading shader part sucessfull
                addShader(code,type);
                content = "";
                code.clear();
            }

        }else if(line.compare("##vertex")==0){
            status = (status==STATUS_START)?STATUS_READING:STATUS_ERROR;
            type = GL_VERTEX_SHADER;

        }else if(line.compare("##fragment")==0){
            status = (status==STATUS_START)?STATUS_READING:STATUS_ERROR;
            type = GL_FRAGMENT_SHADER;

        }else if(line.compare("##geometry")==0){
            status = (status==STATUS_START)?STATUS_READING:STATUS_ERROR;
            type = GL_GEOMETRY_SHADER;
        }else if(status == STATUS_READING){
            //normal code line
            code.push_back(line+'\n');
        }



        if(status == STATUS_ERROR){
            std::cerr<<"Shader-Loader: Error "<<errorMsg<<" in line "<<lineCount<<"\n";
            return false;
        }
    }

    createProgram();
    return true;
}



GLuint Shader::createProgram(){

    program = glCreateProgram();

    //attach all shaders
    for(ShaderPart& sp : shaders){
        glAttachShader(program,sp.id);
    }



    glLinkProgram(program);



    printProgramLog(program);

    for(ShaderPart& sp : shaders){
        sp.deleteGLShader();
    }

    Error::quitWhenError("Shader::createProgram after link");

    checkUniforms();
    return program;
}


void Shader::addShader(std::vector<std::string>& content, GLenum type){

//    addInjectionsToCode(type,content);


    ShaderPart shader;
    shader.code = content;
    shader.type = type;
    shader.addInjections(injections);

    shader.createGLShader();
    if(shader.compile()){
        shaders.push_back(shader);
    }

    Error::quitWhenError("Shader::addShader");

}

void Shader::addShaderFromFile(const std::string &file, GLenum type){
    cout<<"Shader-Loader: Reading file "<<file<<"\n";
    std::string content;


    std::vector<std::string> data = loadAndPreproccess(file);

    if(data.size()<=0)
        return;


    for(std::string line : data){
        content.append(line);
        content.append("\n");
    }


     addShader(data,type);
}


void Shader::bind(){
    if(program==0){
        std::cerr<<"bind: no shader loaded!\n";
        exit(1);
        return;
    }
    glUseProgram(program);
}

void Shader::unbind(){
    glUseProgram(0);
}

GLint Shader::getUniformLocation(const char* name){
    GLint i = glGetUniformLocation(program,name);
    if(i==-1){
        //        cout<<"Cannot find uniform: "<<name<<endl;
    }
    return i;
}

void Shader::getUniformInfo(GLuint location)
{
    const GLsizei bufSize = 128;

    GLsizei length;
    GLint size;
    GLenum type;
    GLchar name[bufSize];

    glGetActiveUniform(program,location,bufSize,&length,&size,&type,name);

    cout<<"uniform info "<<location<<endl;
    cout<<"name "<<name<<endl;
    //    cout<<"length "<<length<<endl;
    cout<<"size "<<size<<endl;
    cout<<"type "<<type<<endl;
}

GLuint Shader::getUniformBlockLocation(const char *name)
{
    GLuint blockIndex = glGetUniformBlockIndex(program, name);

    if(blockIndex==GL_INVALID_INDEX){
        std::cerr<<"glGetUniformBlockIndex: uniform block invalid!"<<endl;
    }
    return blockIndex;
}

void Shader::setUniformBlockBinding(GLuint blockLocation, GLuint bindingPoint)
{
    glUniformBlockBinding(program, blockLocation, bindingPoint);
}

GLint Shader::getUniformBlockSize(GLuint blockLocation)
{
    GLint ret;
    glGetActiveUniformBlockiv(program,blockLocation,GL_UNIFORM_BLOCK_DATA_SIZE,&ret);
    return ret;
}

std::vector<GLint> Shader::getUniformBlockIndices(GLuint blockLocation)
{
    GLint ret;
    glGetActiveUniformBlockiv(program,blockLocation,GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS,&ret);

    std::vector<GLint> indices(ret);
    glGetActiveUniformBlockiv(program,blockLocation,GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES,&indices[0]);

    return indices;
}

std::vector<GLint> Shader::getUniformBlockSize(GLuint blockLocation, std::vector<GLint> indices)
{
    std::vector<GLint> ret(indices.size());
    glGetActiveUniformsiv(program,indices.size(),(GLuint*)indices.data(),GL_UNIFORM_SIZE,ret.data());
    return ret;
}

std::vector<GLint> Shader::getUniformBlockType(GLuint blockLocation, std::vector<GLint> indices)
{
    std::vector<GLint> ret(indices.size());
    glGetActiveUniformsiv(program,indices.size(),(GLuint*)indices.data(),GL_UNIFORM_TYPE,ret.data());
    return ret;
}

std::vector<GLint> Shader::getUniformBlockOffset(GLuint blockLocation, std::vector<GLint> indices)
{
    std::vector<GLint> ret(indices.size());
    glGetActiveUniformsiv(program,indices.size(),(GLuint*)indices.data(),GL_UNIFORM_OFFSET,ret.data());
    return ret;
}


void Shader::printProgramLog( GLuint program ){
    //Make sure name is shader
    if( glIsProgram( program ) == GL_TRUE )
    {
        //Program log length
        int infoLogLength = 0;
        int maxLength = infoLogLength;

        //Get info std::string length
        glGetProgramiv( program, GL_INFO_LOG_LENGTH, &maxLength );

        //Allocate std::string
        char* infoLog = new char[ maxLength ];

        //Get info log
        glGetProgramInfoLog( program, maxLength, &infoLogLength, infoLog );
        if( infoLogLength > 0 )
        {
            //Print Log
            std::cout<<"program error:"<<std::endl;
            std::cout<<  infoLog << std::endl;
        }

        //Deallocate std::string
        delete[] infoLog;
    }
    else
    {
        cout<< "Name "<<program<<" is not a program"<<endl;
    }
}



void Shader::upload(int location, const mat4 &m){
    glUniformMatrix4fv(location,1,GL_FALSE, (GLfloat*)&m[0]);
}

void Shader::upload(int location, const vec4 &v){
    glUniform4fv(location,1,(GLfloat*)&v[0]);
}

void Shader::upload(int location, const vec3 &v){
    glUniform3fv(location,1,(GLfloat*)&v[0]);
}

void Shader::upload(int location, const vec2 &v){
    glUniform2fv(location,1,(GLfloat*)&v[0]);
}

void Shader::upload(int location, const int &i){
    glUniform1i(location,(GLint)i);
}

void Shader::upload(int location, const float &f){
    glUniform1f(location,(GLfloat)f);
}


void Shader::upload(int location, int count, mat4* m){
    glUniformMatrix4fv(location,count,GL_FALSE,(GLfloat*)m);
}

void Shader::upload(int location, int count, vec4* v){
    glUniform4fv(location,count,(GLfloat*)v);
}

void Shader::upload(int location, int count, vec3* v){
    glUniform3fv(location,count,(GLfloat*)v);
}

void Shader::upload(int location, int count, vec2* v){
    glUniform2fv(location,count,(GLfloat*)v);
}


