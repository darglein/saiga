#include "saiga/text/TextOverlay2D.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/text/text.h"

#include <algorithm>

TextOverlay2D::TextOverlay2D(int width, int height):width(width),height(height){
    proj = glm::ortho(0.0f,(float)width,0.0f,(float)height,1.0f,-1.0f);

}

void TextOverlay2D::render(){
    textShader->bind();
    textShader->uploadProj(proj);
    for(Text* &text : texts){
        if(text->visible)
            text->draw(textShader);
    }
    textShader->unbind();
}

void TextOverlay2D::addText(Text* text){
    texts.push_back(text);
}

void TextOverlay2D::removeText(Text *text)
{
   texts.erase( std::remove(texts.begin(),texts.end(),text),texts.end());
}

void TextOverlay2D::setTextShader(TextShader* textShader){
    textShader->bind();
    textShader->uploadProj(proj);
    textShader->unbind();
    this->textShader= textShader;
}

