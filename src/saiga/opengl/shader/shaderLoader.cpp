/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/shader/shaderLoader.h"

namespace Saiga
{
std::shared_ptr<Shader> ShaderLoader::loadFromFile(const std::string& name,
                                                   const ShaderPart::ShaderCodeInjections& params)
{
    (void)params;
    cout << "fail ShaderLoader::loadFromFile " << name << endl;

    SAIGA_ASSERT(0);

    return nullptr;
}

void ShaderLoader::reload()
{
    cout << "ShaderLoader::reload " << objects.size() << endl;
    for (auto& object : objects)
    {
        auto name   = std::get<0>(object);
        auto sci    = std::get<1>(object);
        auto shader = std::get<2>(object);


        std::string fullName = shaderPathes.getFile(name);
        auto ret             = reload(shader, fullName, sci);
        SAIGA_ASSERT(ret);
    }
}

bool ShaderLoader::reload(std::shared_ptr<Shader> shader, const std::string& name,
                          const ShaderPart::ShaderCodeInjections& sci)
{
    cout << "ShaderLoader::reload " << name << endl;
    ShaderPartLoader spl(name, sci);
    if (spl.load())
    {
        spl.reloadShader(shader);
        return true;
    }
    return false;
}

}  // namespace Saiga
