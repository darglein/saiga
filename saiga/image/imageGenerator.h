#pragma once

#include "saiga/util/glm.h"
#include "saiga/image/image.h"


class SAIGA_GLOBAL ImageGenerator{
public:
    std::shared_ptr<int> p;


    /**
     * Creates a checker board textures from 2 colors.
     * quadSize is the size in pixels of quad.
     * The total size of the texture is (numQuadsX*quadSize,numQuadsY*quadSize)
     */

    static std::shared_ptr<Image> checkerBoard(vec3 color1, vec3 color2, int quadSize, int numQuadsX, int numQuadsY);


    /**
     * Creates a random rbga Image in the range [0,1].
     */
    static std::shared_ptr<Image> randomNormalized(int width, int height);
};
