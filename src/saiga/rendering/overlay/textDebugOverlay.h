/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/util/tostring.h"
#include "saiga/util/glm.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/rendering/object3d.h"
#include "saiga/text/TextOverlay2D.h"
#include "saiga/rendering/overlay/Layout.h"
#include "saiga/text/text.h"
#include <vector>

namespace Saiga {

#ifdef SAIGA_USE_FREETYPE

class TextureAtlas;

class Text;

class SAIGA_GLOBAL TextDebugOverlay {
public:
    class TDOEntry{
    public:
        Text* text;
        int valueIndex;
    };


    float borderX = 0.01f;
    float borderY = 0.05f;

    float paddingY = 0.002f;
    float textSize = 0.04f;

    TextParameters textParameters;



    TextOverlay2D overlay;
    TextureAtlas* textureAtlas;

    Text* text;

    Layout layout;

    std::vector<TDOEntry> entries;

    TextDebugOverlay(int w, int h);
    ~TextDebugOverlay();

    void init(TextureAtlas* textureAtlas);
    void render();

    int createItem(const std::string& name);

    template<typename T>
    void updateEntry(int id,const T& v);


};


template<typename T>
void TextDebugOverlay::updateEntry(int id,const T& v)
{
        entries[id].text->updateText(to_string(v),entries[id].valueIndex);
}

#endif

}
