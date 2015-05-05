#include "text/TextOverlay3D.h"
#include "libhello/opengl/shader.h"
#include "libhello/text/text.h"

TextOverlay3D::TextOverlay3D(){

}

void TextOverlay3D::render(Camera *cam){
    renderText(cam);
}

void TextOverlay3D::addText(Text* text){
    texts.push_back(text);
}

void TextOverlay3D::setTextShader(TextShader* textShader){

    this->textShader= textShader;
}

void TextOverlay3D::renderText(Camera *cam){
    textShader->bind();

    textShader->uploadProj(cam->proj);
    textShader->uploadView(cam->view);

    mat4 v = cam->model;
    v[3] = vec4(0,0,0,1);


    for(Text* &text : texts){

        //make this text face towards the camera
        text->calculateModel();
        text->model =  text->model * v;

        text->draw(textShader);
    }
    textShader->unbind();
}
