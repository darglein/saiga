#include "saiga/rendering/overlay/textDebugOverlay.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/geometry/triangle_mesh.h"
#include "saiga/opengl/framebuffer.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/text/textureAtlas.h"
#include "saiga/text/textShader.h"
#include "saiga/text/text.h"

TextDebugOverlay::TextDebugOverlay(int w, int h): overlay(1,1),layout(w,h){

}

TextDebugOverlay::~TextDebugOverlay()
{
    for(TDOEntry &entry : entries){
        delete entry.text;
    }
}

void TextDebugOverlay::init(TextureAtlas *textureAtlas)
{
    this->textureAtlas = textureAtlas;


}

void TextDebugOverlay::render()
{
    overlay.render();
}

int TextDebugOverlay::createItem(const std::string &name, int valueChars)
{
    int id = entries.size();
    int length = name.size() + valueChars;
    TDOEntry entry;
    entry.length = length;
    entry.valueIndex = name.size();

    entry.text = new Text(textureAtlas,"");
//    entry.text->color = vec4(1,0,0,1);
//    entry.text->params.setColor(vec4(1,0,0,1),0.1f);
    entry.text->params = textParameters;
//    entry.text->strokeColor = vec4(0.1f,0.1f,0.1f,1.0f);
    overlay.addText(entry.text);


//    textGenerator->updateText(entry.text,name+std::string(valueChars,' '),0);
    entry.text->updateText(name+std::string(valueChars,'\0'),0);
    aabb bb = entry.text->getAabb();
    bb.growBox(textureAtlas->getMaxCharacter());


    int y = id;

    vec2 relPos(0);
    relPos.x = borderX;
    relPos.y =  1.0f-((y) * (paddingY+textSize) + borderY);

    layout.transform(entry.text,bb,relPos,textSize,Layout::LEFT,Layout::RIGHT);

    entries.push_back(entry);


    entry.text->updateText(std::to_string(123),entry.valueIndex);

    return id;
}

template<>
void TextDebugOverlay::updateEntry<std::string>(int id, std::string v)
{
    entries[id].text->updateText(v,entries[id].valueIndex);
}


