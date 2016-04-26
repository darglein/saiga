#pragma once

#include "saiga/opengl/opengl.h"
#include "saiga/util/glm.h"

#include <vector>

/**
 * @brief The CodeInjection class
 * The code injections are added to the top of the shader.
 * This can for exapmple be used to add different #define.
 */
class SAIGA_GLOBAL ShaderCodeInjection{
public:
	//shader type, must be one of the supported types
	GLenum type;
	std::string code;
	int line;

	ShaderCodeInjection() :type(GL_INVALID_ENUM), code(""), line(0){}
	ShaderCodeInjection(GLenum type, const std::string &code, int line) :type(type), code(code), line(line){}
};

SAIGA_GLOBAL inline bool operator==(const ShaderCodeInjection& lhs, const ShaderCodeInjection& rhs) {
	return lhs.type == rhs.type && lhs.code == rhs.code && lhs.line == rhs.line;
}

/**
 * The ShaderPart class represents an actual Shader Object in OpenGL while the
 * Shader class represents a program.
 */

class SAIGA_GLOBAL ShaderPart{
public:
	typedef std::vector<ShaderCodeInjection> ShaderCodeInjections;

	//supported shader types:
	//GL_COMPUTE_SHADER, GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER, GL_TESS_EVALUATION_SHADER, GL_GEOMETRY_SHADER,  GL_FRAGMENT_SHADER
	static const GLenum shaderTypes[];
	static const std::string shaderTypeStrings[];
	static const int shaderTypeCount = 6;

	GLenum type;
	std::vector<std::string> code;
	std::string error = "";
	GLint id = 0;

	ShaderPart();
	~ShaderPart();

	void createGLShader();
	void deleteGLShader();

	/**
	* writes the complete code (with shader injections) to a file
	* if there was a error compiling it also writes the error to file+"error.txt"
	*
	* usefull for debugging shaders with alot of includes, to better understand errors.
	*/
	bool writeToFile(const std::string& file);

	bool compile();
	void printShaderLog();
	void printError();

	std::string typeToName(GLenum type);
	std::string getTypeString();

	void addInjection(const ShaderCodeInjection &sci);
	void addInjections(const ShaderCodeInjections &scis);
};
