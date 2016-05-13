#include "saiga/opengl/shader/shaderpart.h"

#include "saiga/util/error.h"
#include <fstream>

const GLenum ShaderPart::shaderTypes[] = {GL_COMPUTE_SHADER, GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER, GL_TESS_EVALUATION_SHADER, GL_GEOMETRY_SHADER,  GL_FRAGMENT_SHADER};
const std::string ShaderPart::shaderTypeStrings[] = {"GL_COMPUTE_SHADER", "GL_VERTEX_SHADER", "GL_TESS_CONTROL_SHADER", "GL_TESS_EVALUATION_SHADER", "GL_GEOMETRY_SHADER",  "GL_FRAGMENT_SHADER"};


ShaderPart::ShaderPart()
{

}

ShaderPart::~ShaderPart()
{
    deleteGLShader();
}

void ShaderPart::createGLShader()
{
    deleteGLShader(); //delete shader if exists
    id = glCreateShader(type);
    if(id==0){
        cout<<"Could not create shader of type: "<<typeToName(type)<<endl;
    }
    assert_no_glerror();
}

void ShaderPart::deleteGLShader()
{
    if(id!=0){
        glDeleteShader(id);
        id = 0;
    }
}

bool ShaderPart::writeToFile(const std::string& file){
	std::cout << "Writing shader to file: " << file << std::endl;
	std::fstream stream;
	stream.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	try {
		stream.open(file, std::fstream::out);
		for (auto str : code){
			stream << str;
		}
		stream.close();
	}
	catch (const std::fstream::failure &e) {
		std::cout << e.what() << std::endl;
		std::cout << "Exception opening/reading file\n";
		return false;
	}

	if (error == ""){
		return true;
	}

	std::string errorFile = file + ".error.txt";
	std::cout << "Writing shader error to file: " << errorFile << std::endl;

	try {
		stream.open(errorFile, std::fstream::out);
		stream << error << endl;
		stream.close();
	}
	catch (const std::fstream::failure &e) {
		std::cout << e.what() << std::endl;
		std::cout << "Exception opening/reading file\n";
		return false;
	}

	return true;
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

    assert_no_glerror();
    //checking compile status
    GLint result = 0;
    glGetShaderiv(id, GL_COMPILE_STATUS, &result);
    if(result == static_cast<GLint>(GL_FALSE) ){
        printShaderLog();
        //TODO: print warnings
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
		this->error = std::string(infoLog);
		printError();
    }

    delete[] infoLog;
}

void ShaderPart::printError()
{
	//no real parsing is done here because different drivers produce completly different messages
	std::cout << this->getTypeString() << " compilation failed. Error:" << std::endl;
	std::cout << error << std::endl;
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

std::string ShaderPart::getTypeString(){
	int i = 0;
	for (; i < shaderTypeCount; ++i){
		if (shaderTypes[i] == type)
			break;
	}
	return shaderTypeStrings[i];
}
