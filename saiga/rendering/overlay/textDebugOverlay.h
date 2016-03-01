#pragma once
#include <sstream>

#include "saiga/util/glm.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/rendering/object3d.h"

#include "saiga/text/TextOverlay2D.h"
#include "saiga/rendering/overlay/Layout.h"
#include <vector>
#include "saiga/text/text.h"

class TextureAtlas;

class Text;

class SAIGA_GLOBAL TextDebugOverlay {
public:
    class TDOEntry{
    public:
        Text* text;
        int length;
        int valueIndex;
    };


    float borderX = 0.01f;
    float borderY = 0.05f;

    float paddingY = 0.001f;
    float textSize = 0.02f;



    TextOverlay2D overlay;
    TextureAtlas* textureAtlas;

    Text* text;

    Layout layout;

    std::vector<TDOEntry> entries;

    TextDebugOverlay();

    void init(TextureAtlas* textureAtlas);
    void render();

    int createItem(const std::string& name, int valueChars);

    template<typename T>
    void updateEntry(int id, T v);


};

template<>
void TextDebugOverlay::updateEntry<std::string>(int id, std::string v);

template<typename T>
void TextDebugOverlay::updateEntry(int id, T v)
{
    std::stringstream sstream;

    sstream << v;
//    textGenerator->updateText(entries[id].text,sstream.str()+"                                  ",entries[id].valueIndex);
    entries[id].text->updateText(sstream.str()+"                                  ",entries[id].valueIndex);
}






