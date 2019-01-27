/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/texture/texture.h"
#include "saiga/util/ObjectCache.h"
#include "saiga/util/singleton.h"

namespace Saiga
{
struct SAIGA_GLOBAL TextureParameters
{
    bool srgb = true;
};

SAIGA_GLOBAL bool operator==(const TextureParameters& lhs, const TextureParameters& rhs);


class SAIGA_GLOBAL TextureLoader : public Singleton<TextureLoader>
{
    friend class Singleton<TextureLoader>;

    ObjectCache<std::string, std::shared_ptr<Texture>, TextureParameters> cache;

   public:
    std::shared_ptr<Texture> load(const std::string& name, const TextureParameters& params = TextureParameters());
};

}  // namespace Saiga
