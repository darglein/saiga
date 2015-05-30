#include "opengl/shader.h"

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
    if(vertShader){
        glDeleteShader(vertShader);
        vertShader = 0;
    }
    if(geoShader){
        glDeleteShader(geoShader);
        geoShader = 0;
    }
    if(fragShader){
        glDeleteShader(fragShader);
        fragShader = 0;
    }
}

Shader::Shader(const string &multi_file) : shaderPath(multi_file),program(0),vertShader(0),geoShader(0),fragShader(0){

}


bool Shader::reload(){
    if(shaderPath.length()<=0){
        cerr<<"Reload only works if the Shader object is created with a multishader in the constructor"<<endl;
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

    const string include("#include ");

    while(!fileStream.eof()) {
        string line;
        std::getline(fileStream, line);

        if(include.size()<line.size() && line.compare(0, include.length(), include)==0){
            line = line.substr(include.size()-1);

            auto it = std::remove(line.begin(),line.end(),'"');
            line.erase(it,line.end());

            it = std::remove(line.begin(),line.end(),' ');
            line.erase(it,line.end());


            //            cout<<"found include ("<<line<<")"<<endl;
            cout<<"including file "<<prefix<<"/"<<line<<endl;

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

bool Shader::addMultiShaderFromFile(const string &multi_file) {

    std::string content,errorMsg;


    std::vector<string> data = loadAndPreproccess(multi_file);

    if(data.size()<=0)
        return false;

    int status = STATUS_WAITING;
    int type = -1;
    int lineCount =0;

    for(string line : data){
        //        std::getline(fileStream, line);
        lineCount++;
        if(line.compare("##start")==0){
            status = (status==STATUS_WAITING)?STATUS_START:STATUS_ERROR;

        }else if(line.compare("##end")==0){
            status = (status==STATUS_READING)?STATUS_WAITING:STATUS_ERROR;

            if(status != STATUS_ERROR){
                //reading shader part sucessfull
                addShader(content.c_str(),type);
                content = "";
                for(int i=0;i<lineCount-1;i++)
                    content.append("\n");
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
        }



        if(status == STATUS_ERROR){
            std::cerr<<"Shader-Loader: Error "<<errorMsg<<" in line "<<lineCount<<"\n";
            return false;
        }else if(status == STATUS_READING){
            content.append(line);
        }
        content.append("\n");
    }

    //    fileStream.close();
    createProgram();
    return true;
}



GLuint Shader::createProgram(){

    program = glCreateProgram();

    if(vertShader)
        glAttachShader(program, vertShader);
    if(geoShader){
        glAttachShader(program, geoShader);
        //TODO disable this for NVIDIA nsight debugging
        glProgramParameteriEXT(program,GL_GEOMETRY_INPUT_TYPE_EXT,GL_TRIANGLES);
        glProgramParameteriEXT(program,GL_GEOMETRY_OUTPUT_TYPE_EXT,GL_TRIANGLE_STRIP);

    }
    if(fragShader)
        glAttachShader(program, fragShader);

    glLinkProgram(program);

    printProgramLog(program);


    if(vertShader){
        glDeleteShader(vertShader);
        vertShader = 0;
    }
    if(geoShader){
        glDeleteShader(geoShader);
        geoShader = 0;
    }
    if(fragShader){
        glDeleteShader(fragShader);
        fragShader = 0;
    }

    checkUniforms();
    return program;
}

GLuint Shader::addShader(const char* content, int type){
    GLuint id = glCreateShader(type);


    GLint result = GL_FALSE;
    // Compile vertex shader
    glShaderSource(id, 1, &content, NULL);
    glCompileShader(id);
    // Check vertex shader
    glGetShaderiv(id, GL_COMPILE_STATUS, &result);

    printShaderLog(id);


    if(!result){
        glDeleteShader(id);
        id = 0;
    }

    switch(type){
    case GL_VERTEX_SHADER:
        vertShader = id;
        break;
    case GL_GEOMETRY_SHADER:
        geoShader = id;
        break;
    case GL_FRAGMENT_SHADER:
        fragShader = id;
        break;

    }

    return id;
}

GLuint Shader::addShaderFromFile(const char* file, int type){
    cout<<"Shader-Loader: Reading file "<<file<<"\n";
    std::string content;


    std::vector<string> data = loadAndPreproccess(file);

    if(data.size()<=0)
        return false;


    for(string line : data){
        content.append(line);
        content.append("\n");
    }


    return addShader(content.c_str(),type);
}

string Shader::typeToName(int type){
    switch(type){
    case GL_VERTEX_SHADER:
        return "Vertex Shader";
    case GL_GEOMETRY_SHADER:
        return "Geometry Shader";
    case GL_FRAGMENT_SHADER:
        return "Fragment Shader";
    default:
        return "Unkown Shader type!";
    }
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
        cerr<<"glGetUniformBlockIndex: uniform block invalid!"<<endl;
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
    if( glIsProgram( program ) )
    {
        //Program log length
        int infoLogLength = 0;
        int maxLength = infoLogLength;

        //Get info string length
        glGetProgramiv( program, GL_INFO_LOG_LENGTH, &maxLength );

        //Allocate string
        char* infoLog = new char[ maxLength ];

        //Get info log
        glGetProgramInfoLog( program, maxLength, &infoLogLength, infoLog );
        if( infoLogLength > 0 )
        {
            //Print Log
            printf( "%s\n", infoLog );
        }

        //Deallocate string
        delete[] infoLog;
    }
    else
    {
        printf( "Name %d is not a program\n", program );
    }
}

void Shader::printShaderLog( GLuint shader ){
    //Make sure name is shader
    if( glIsShader( shader ) )
    {
        //Shader log length
        int infoLogLength = 0;
        int maxLength = infoLogLength;

        //Get info string length
        glGetShaderiv( shader, GL_INFO_LOG_LENGTH, &maxLength );

        //Allocate string
        char* infoLog = new char[ maxLength ];

        //Get info log
        glGetShaderInfoLog( shader, maxLength, &infoLogLength, infoLog );
        if( infoLogLength > 0 )
        {
            //Print Log
            printf( "%s\n", infoLog );
        }

        //Deallocate string
        delete[] infoLog;
    }
    else
    {
        printf( "Name %d is not a shader\n", shader );
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


Shader* ShaderLoader::loadFromFile(const std::string &name){
    Shader* shader = new Shader(name);
    if(shader->reload()){
        return shader;
    }
    delete shader;
    return nullptr;
}

void ShaderLoader::reload(){
    for(Shader* &object : objects){
        object->reload();
    }
}
