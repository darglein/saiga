#include "text/TextOverlay2D.h"
#include "libhello/opengl/shader.h"
#include "libhello/text/text.h"

TextOverlay2D::TextOverlay2D(int width, int height):width(width),height(height){
    proj = glm::ortho(0.0f,(float)width,0.0f,(float)height,1.0f,-1.0f);

}

void TextOverlay2D::render(){
    renderText();
}

void TextOverlay2D::addText(Text* text){
    texts.push_back(text);
}

void TextOverlay2D::setTextShader(TextShader* textShader){
    textShader->bind();
    textShader->uploadProj(proj);
    textShader->unbind();
    this->textShader= textShader;
}

void TextOverlay2D::renderText(){
    textShader->bind();
    textShader->uploadProj(proj);
    for(Text* &text : texts){
        text->draw(textShader);
    }
    textShader->unbind();
}
