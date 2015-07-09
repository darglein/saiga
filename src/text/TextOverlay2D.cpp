#include "libhello/text/TextOverlay2D.h"
#include "libhello/opengl/basic_shaders.h"
#include "libhello/text/text.h"

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

void TextOverlay2D::setTextShader(TextShader* textShader){
//    cout<<"set text shader "<<textShader<<endl;

    textShader->bind();
    textShader->uploadProj(proj);
    textShader->unbind();
    this->textShader= textShader;
}

