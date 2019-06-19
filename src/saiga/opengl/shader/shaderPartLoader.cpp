/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/shader/shaderPartLoader.h"

#include "saiga/opengl/error.h"
#include "saiga/opengl/shader/shader.h"
#include "saiga/core/util/fileChecker.h"

#include <algorithm>
#include <fstream>
#include <regex>

namespace Saiga
{
#define STATUS_WAITING 0
#define STATUS_READING 1


bool ShaderPartLoader::addLineDirectives = false;

ShaderPartLoader::ShaderPartLoader() : ShaderPartLoader("", ShaderCodeInjections()) {}

ShaderPartLoader::ShaderPartLoader(const std::string& file, const ShaderCodeInjections& injections)
    : file(file), injections(injections)
{
}

ShaderPartLoader::~ShaderPartLoader() {}



bool ShaderPartLoader::load()
{
    std::vector<std::string> data;

    if (!loadAndPreproccess(file, data)) return false;


    std::vector<std::string> code;
    int status    = STATUS_WAITING;
    GLenum type   = GL_INVALID_ENUM;
    int lineCount = 0;


    // https://de.wikipedia.org/wiki/Byte_Order_Mark
    // quick check for utf8-BOM
    if (data.size() > 0)
    {
        std::string& line = data[0];
        if (line.size() >= 3)
        {
            if ((unsigned char)line[0] == 0xEF && (unsigned char)line[1] == 0xBB && (unsigned char)line[2] == 0xBF)
            {
                line = line.substr(3);
            }
        }
    }

    for (std::string line : data)
    {
        lineCount++;

        bool readLine = true;
        for (int i = 0; i < ShaderPart::shaderTypeCount; ++i)
        {
            std::string key("##" + ShaderPart::shaderTypeStrings[i]);
            // this only compares the first characteres of line, so that for example addittional '\r's are ignored.
            if (line.compare(0, key.size(), key) == 0)
            {
                //                std::cout << "found key " << key << std::endl;
                if (status == STATUS_READING)
                {
                    addShader(code, type);
                    code.clear();
                }
                status   = STATUS_READING;
                type     = ShaderPart::shaderTypes[i];
                readLine = false;
                break;
            }
        }

        if (status == STATUS_READING && readLine)
        {
            code.push_back(line + '\n');
        }
    }

    if (status == STATUS_READING)
    {
        addShader(code, type);
        code.clear();
    }


    return true;
}

inline std::string getFileFromInclude(const std::string& file, std::string line)
{
    const std::string include("#include ");
    line = line.substr(include.size() - 1);

    auto it = std::remove(line.begin(), line.end(), '"');
    line.erase(it, line.end());

    it = std::remove(line.begin(), line.end(), ' ');
    line.erase(it, line.end());

    // recursivly load includes
    std::string includeFileName = line;
    includeFileName             = SearchPathes::shader.getRelative(file, includeFileName);
    return includeFileName;
}

bool ShaderPartLoader::loadAndPreproccess(const std::string& file, std::vector<std::string>& ret)
{
    std::ifstream fileStream(file, std::ios::in);
    if (!fileStream.is_open())
    {
        return false;
    }

    const std::string version("#version");
    const std::string include("#include ");

    // TODO: parse with regex. Requires gcc 4.9+

    // first pass:
    // 1. read file line by line and save it into ret vector
    // 2. add #line commands after #version and before and after #includes
    int addedLines = 0;  // count number of added "line" commands.
    while (!fileStream.eof())
    {
        std::string line;
        std::getline(fileStream, line);

        // remove carriage return from windows
        line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());

        if (include.size() < line.size() && line.compare(0, include.length(), include) == 0)
        {
            std::string includeFileName = getFileFromInclude(file, line);

            if (addLineDirectives)
            {
                // add #line before and after #includes
                std::string lineCommand = "#line " + std::to_string(1) + " \"" + includeFileName + "\"";
                ret.push_back(lineCommand);
                addedLines++;
            }
            ret.push_back(line);
            if (addLineDirectives)
            {
                std::string lineCommand = "#line " + std::to_string(ret.size() - addedLines + 1) + " \"" + file + "\"";
                ret.push_back(lineCommand);
                addedLines++;
            }
        }
        else if (version.size() < line.size() && line.compare(0, version.length(), version) == 0)
        {
            // add a #line command after the #version command
            ret.push_back(line);
            if (addLineDirectives)
            {
                std::string lineCommand = "#line " + std::to_string(ret.size() - addedLines + 1) + " \"" + file + "\"";
                ret.push_back(lineCommand);
                addedLines++;
            }
        }
        else
        {
            ret.push_back(line);
        }
    }

    // second pass:
    // loop over vector and replace #include commands with the actual code

    for (unsigned int i = 0; i < ret.size(); ++i)
    {
        std::string line = ret[i];
        if (include.size() < line.size() && line.compare(0, include.length(), include) == 0)
        {
            std::string includeFileName = getFileFromInclude(file, line);

            std::vector<std::string> tmp;
            if (!loadAndPreproccess(includeFileName, tmp))
            {
                std::cerr << "ShaderPartLoader: Could not open included file: " << line << std::endl;
                std::cerr << "Extracted filename: '" << includeFileName << "'" << std::endl;
                std::cerr << "Basefile: " << file << std::endl;
                std::cerr << "Make sure it exists and the search pathes are set." << std::endl;
                SAIGA_ASSERT(0);
            }
            ret.erase(ret.begin() + i);
            ret.insert(ret.begin() + i, tmp.begin(), tmp.end());
        }
    }
    return true;
}

void ShaderPartLoader::addShader(std::vector<std::string>& content, GLenum type)
{
    auto shader  = std::make_shared<ShaderPart>();
    shader->code = content;
    shader->type = type;
    shader->addInjections(injections);

    shader->createGLShader();
    if (shader->compile())
    {
        shaders.push_back(shader);
    }
    else
    {
        FileChecker fc;
        std::string name = fc.getFileName(this->file);
        shader->writeToFile("debug/" + name);
    }

    assert_no_glerror();
}

void ShaderPartLoader::reloadShader(std::shared_ptr<Shader> shader)
{
    //    std::cout<<"ShaderPartLoader::reloadShader"<<endl;
    shader->destroyProgram();

    shader->shaders = shaders;
    shader->createProgram();

    std::cout << "Loaded: " << file << " ( ";
    for (auto& sp : shaders)
    {
        std::cout << sp->type << " ";
    }
    std::cout << ")" << std::endl;

    assert_no_glerror();
}

}  // namespace Saiga
