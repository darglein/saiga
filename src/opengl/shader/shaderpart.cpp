#include "saiga/opengl/shader/shaderpart.h"

#include "saiga/util/error.h"


void ShaderPart::createGLShader()
{
    deleteGLShader(); //delete shader if exists
    id = glCreateShader(type);
    if(id==0){
        cout<<"Could not create shader of type: "<<typeToName(type)<<endl;
    }
    Error::quitWhenError("ShaderPart::createGLShader");
}

void ShaderPart::deleteGLShader()
{
    if(id!=0){
        glDeleteShader(id);
        id = 0;
    }
}

bool ShaderPart::compile()
{
    std::string data;
    for(std::string line : code){
        data.append(line);
    }
    const GLchar* str = data.c_str();
    glShaderSource(id, 1,&str , 0);
    glCompileShader(id);
    //checking compile status
    GLint result = 0;
    glGetShaderiv(id, GL_COMPILE_STATUS, &result);
    if(result == static_cast<GLint>(GL_FALSE) ){
        printShaderLog();
        return false;
    }
    return true;
}

void ShaderPart::printShaderLog()
{

    int infoLogLength = 0;
    int maxLength = infoLogLength;

    glGetShaderiv( id, GL_INFO_LOG_LENGTH, &maxLength );

    GLchar* infoLog = new GLchar[ maxLength ];

    glGetShaderInfoLog( id, maxLength, &infoLogLength, infoLog );
    if( infoLogLength > 0 ){
        parseShaderError(std::string(infoLog));
    }

    delete[] infoLog;
}


void ShaderPart::addInjection(const ShaderCodeInjection& sci)
{
    std::string injection;
    if(sci.type==type){
        injection =  sci.code+ '\n' ;
        code.insert(code.begin()+sci.line,injection);
    }


}

void ShaderPart::addInjections(const ShaderPart::ShaderCodeInjections &scis)
{
    for(const ShaderCodeInjection& sci : scis){
        addInjection(sci);
    }
}


void ShaderPart::parseShaderError(const std::string &message)
{
    //no real parsing is done here because different drivers produce completly different messages
    std::cout<<"shader error:"<<std::endl;
    std::cout<< message << std::endl;
}


std::string ShaderPart::typeToName(GLenum type){
    switch(type){
    case GL_VERTEX_SHADER:
        return "Vertex Shader";
    case GL_GEOMETRY_SHADER:
        return "Geometry Shader";
    case GL_FRAGMENT_SHADER:
        return "Fragment Shader";
    default:
        return "Unkown Shader type! ";
    }
}
