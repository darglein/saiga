/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/shader/shaderPartLoader.h"

#include "saiga/core/util/fileChecker.h"
#include "saiga/opengl/error.h"
#include "saiga/opengl/shader/shader.h"

#include <algorithm>
#include <fstream>
#include <regex>

namespace Saiga
{

bool ShaderPartLoader::addLineDirectives = false;

ShaderPartLoader::ShaderPartLoader(const std::string& file, const ShaderCodeInjections& injections)
    : file(file), injections(injections)
{
}


bool ShaderPartLoader::load()
{
    auto c = LoadFileAndResolveIncludes(file, addLineDirectives);
    if (!c.valid) return false;

    std::cout << "file " << file << std::endl;
    for (auto p : c.parts)
    {
        if (p.type.empty() || p.end - p.start == 0) continue;
        GLenum gl_type = GL_NONE;
        for (int i = 0; i < ShaderPart::shaderTypeCount; ++i)
        {
            if (p.type == ShaderPart::shaderTypeStrings[i])
            {
                gl_type = ShaderPart::shaderTypes[i];
            }
        }
        SAIGA_ASSERT(gl_type != GL_NONE, "Unknown shader type: " + p.type);
        std::vector<std::string> content(c.code.begin() + p.start, c.code.begin() + p.end);
        addShader(content, gl_type);
    }
    return true;
}

void ShaderPartLoader::addShader(std::vector<std::string>& content, GLenum type)
{
#if 0
    std::cout << "loading shader part " << type << std::endl;
    for (int i = 0; i < content.size(); ++i)
    {
        auto& line = content[i];
        std::cout << std::setw(5) << i << ":" << line << std::endl;
    }
#endif



    auto shader = std::make_shared<ShaderPart>(content, type, injections);
    if (shader->valid)
    {
        shaders.push_back(shader);
    }
    assert_no_glerror();
}

void ShaderPartLoader::reloadShader(std::shared_ptr<Shader> shader)
{
    //    std::cout<<"ShaderPartLoader::reloadShader"<<endl;
    shader->destroyProgram();

    shader->shaders = shaders;
    shader->createProgram();

    std::string type_str;
    for (auto& sp : shaders)
    {
        type_str += to_string(sp->type) + " ";
    }
    VLOG(1) << "Loaded: " << file << " ( " << type_str << ") Id=" << shader->program;

    assert_no_glerror();
}

}  // namespace Saiga
