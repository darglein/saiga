/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/texture/texture.h"

#include "saiga/util/loader.h"
#include "saiga/util/singleton.h"

namespace Saiga {

struct SAIGA_GLOBAL TextureParameters{
    bool srgb = true;
};

SAIGA_GLOBAL bool operator==(const TextureParameters& lhs, const TextureParameters& rhs);


class SAIGA_GLOBAL TextureLoader : public Loader<std::shared_ptr<Texture>,TextureParameters>, public Singleton <TextureLoader>{
    friend class Singleton <TextureLoader>;
public:
    std::shared_ptr<Texture> loadFromFile(const std::string &name, const TextureParameters &params);

    std::shared_ptr<Texture> textureFromImage(Image &im, const TextureParameters &params) const;
};

}
