/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/shader/shaderpart.h"

#include "saiga/opengl/error.h"

#include <fstream>
#include <iostream>

namespace Saiga
{
const GLenum ShaderPart::shaderTypes[] = {GL_COMPUTE_SHADER,         GL_VERTEX_SHADER,   GL_TESS_CONTROL_SHADER,
                                          GL_TESS_EVALUATION_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER};
const std::string ShaderPart::shaderTypeStrings[] = {"GL_COMPUTE_SHADER",      "GL_VERTEX_SHADER",
                                                     "GL_TESS_CONTROL_SHADER", "GL_TESS_EVALUATION_SHADER",
                                                     "GL_GEOMETRY_SHADER",     "GL_FRAGMENT_SHADER"};


ShaderPart::ShaderPart(const std::vector<std::string>& content, GLenum type, const ShaderCodeInjections& injections)
{
    code = content;
    this->type = type;
    for (auto& sci : injections)
    {
        if (sci.type == type)
        {
            code.insert(code.begin() + sci.line, sci.code + "\n");
        }
    }

    deleteGLShader();  // delete shader if exists
    id = glCreateShader(type);
    if (id == 0)
    {
        std::cout << "Could not create shader of type: " << typeToName(type) << std::endl;
    }
    assert_no_glerror();

    compile();
}

ShaderPart::~ShaderPart()
{
    deleteGLShader();
}


void ShaderPart::deleteGLShader()
{
    if (id != 0)
    {
        glDeleteShader(id);
        id = 0;
    }
}

bool ShaderPart::writeToFile(const std::string& file)
{
    std::string shaderFile = file + "." + this->getTypeString() + ".txt";
    std::cout << "Writing shader to file: " << shaderFile << std::endl;
    std::fstream stream;
    stream.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try
    {
        stream.open(shaderFile, std::fstream::out);
        for (auto str : code)
        {
            stream << str;
        }
        stream.close();
    }
    catch (const std::fstream::failure& e)
    {
        std::cout << e.what() << std::endl;
        std::cout << "Exception opening/reading file\n";
        return false;
    }

    if (error == "")
    {
        return true;
    }

    std::string errorFile = file + "." + this->getTypeString() + ".error.txt";
    std::cout << "Writing shader error to file: " << errorFile << std::endl;

    try
    {
        stream.open(errorFile, std::fstream::out);
        stream << error << std::endl;
        stream.close();
    }
    catch (const std::fstream::failure& e)
    {
        std::cout << e.what() << std::endl;
        std::cout << "Exception opening/reading file\n";
        return false;
    }

    return true;
}

bool ShaderPart::compile()
{
    std::string data;
    data.reserve(code.size() * 20);
    for (std::string line : code)
    {
        data.append(line);
        data.push_back('\n');
    }
    const GLchar* str = data.c_str();
    glShaderSource(id, 1, &str, 0);
    assert_no_glerror();
    glCompileShader(id);
    assert_no_glerror();

    printShaderLog();

    // checking compile status
    GLint result = 0;
    glGetShaderiv(id, GL_COMPILE_STATUS, &result);
    assert_no_glerror();
    valid = !(result == static_cast<GLint>(GL_FALSE));
    return valid;
}

void ShaderPart::printShaderLog()
{
    int infoLogLength = 0;
    int maxLength     = infoLogLength;

    glGetShaderiv(id, GL_INFO_LOG_LENGTH, &maxLength);

    if (maxLength == 0) return;

    std::vector<GLchar> infoLog(maxLength, 0);
    //    GLchar* infoLog = new GLchar[ maxLength ];

    glGetShaderInfoLog(id, maxLength, &infoLogLength, infoLog.data());
    if (infoLogLength > 0)
    {
        this->error = std::string(infoLog.begin(), infoLog.begin() + infoLogLength);
        printError();
    }
}

void ShaderPart::printError()
{
    // no real parsing is done here because different drivers produce completly different messages
    std::cout << this->getTypeString() << " info log:" << std::endl;
    std::cout << error << std::endl;
}




std::string ShaderPart::typeToName(GLenum type)
{
    switch (type)
    {
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

std::string ShaderPart::getTypeString()
{
    int i = 0;
    for (; i < shaderTypeCount; ++i)
    {
        if (shaderTypes[i] == type) break;
    }
    return shaderTypeStrings[i];
}

}  // namespace Saiga
