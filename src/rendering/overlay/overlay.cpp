#include "rendering/overlay/overlay.h"
#include "libhello/opengl/shader.h"
#include "libhello/text/text.h"

Overlay::Overlay(int width, int height):width(width),height(height){
    proj = glm::ortho(0.0f,(float)width,0.0f,(float)height,1.0f,-1.0f);

}

void Overlay::render(){
    renderText();
}

void Overlay::addText(Text* text){
    texts.push_back(text);
}

void Overlay::setTextShader(TextShader* textShader){
    textShader->bind();
    textShader->uploadProj(proj);
    textShader->unbind();
    this->textShader= textShader;
}

void Overlay::renderText(){
    textShader->bind();
    textShader->uploadProj(proj);
    for(Text* &text : texts){
        text->draw(textShader);
    }
    textShader->unbind();
}
