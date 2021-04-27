/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "saiga/opengl/texture/TextureLoader.h"


namespace Saiga
{
bool operator==(const TextureParameters& lhs, const TextureParameters& rhs)
{
    return std::tie(lhs.srgb) == std::tie(rhs.srgb);
}


std::shared_ptr<Texture> TextureLoader::load(const std::string& name, const TextureParameters& params)
{
    std::string fullName = SearchPathes::image(name);
    if (fullName.empty())
    {
        std::cout << "Could not find file '" << name << "'. Make sure it exists and the search pathes are set." << std::endl;
        std::cerr << SearchPathes::image << std::endl;
        SAIGA_ASSERT(0);
    }

    std::shared_ptr<Texture> object;
    auto inCache = cache.get(fullName, object, params);

    if (inCache)
    {
    }
    else
    {
        bool erg;
        Image im;
        erg = im.load(fullName);


        if (erg)
        {
            object = std::make_shared<Texture>(im, params.srgb, true);
        }

        cache.put(fullName, object, params);
    }
    SAIGA_ASSERT(object);
    return object;
}

}  // namespace Saiga
