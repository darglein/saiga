/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ShaderLoaderHelper.h"

#include "saiga/core/util/tostring.h"

namespace Saiga
{
namespace Vulkan
{
ShaderLoadHelper::EndingType ShaderLoadHelper::getEnding(const std::string& file)
{
    std::string ending = fileEnding(file);

    for (int i = 0; i < 7; ++i)
    {
        if (ending == std::get<1>(fileEndings[i])) return fileEndings[i];
    }

    return fileEndings[7];
}

std::string ShaderLoadHelper::stripEnding(const std::string& file)
{
    return removeFileEnding(file);
}


}  // namespace Vulkan
}  // namespace Saiga
