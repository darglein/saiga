/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/shader/shaderLoader.h"

namespace Saiga
{
ShaderLoader shaderLoader;

void ShaderLoader::reload()
{
    std::cout << "ShaderLoader::reload " << cache.objects.size() << std::endl;
    for (auto& object : cache.objects)
    {
        auto name   = std::get<0>(object);
        auto sci    = std::get<1>(object);
        auto shader = std::get<2>(object);


        std::string fullName = SearchPathes::shader(name);
        auto ret             = reload(shader, fullName, sci);
        SAIGA_ASSERT(ret);
    }
}

bool ShaderLoader::reload(std::shared_ptr<Shader> shader, const std::string& name,
                          const ShaderPart::ShaderCodeInjections& sci)
{
    std::cout << "ShaderLoader::reload " << name << std::endl;
    ShaderPartLoader spl(name, sci);
    if (spl.load())
    {
        spl.reloadShader(shader);
        return true;
    }
    return false;
}



}  // namespace Saiga
