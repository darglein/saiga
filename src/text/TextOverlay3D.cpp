#include "text/TextOverlay3D.h"
#include "libhello/opengl/shader.h"
#include "libhello/text/text.h"
#include <algorithm>

TextOverlay3D::TextOverlay3D(){

}

void TextOverlay3D::render(Camera *cam){
    renderText(cam);
}

void TextOverlay3D::addText(std::unique_ptr<Text> text, float duration){
    texts.push_back(std::make_pair(std::move(text), duration));
}

void TextOverlay3D::update(float secondsPerTick)
{
    texts.erase(std::remove_if(texts.begin(), texts.end(),
                   [secondsPerTick](std::pair<std::unique_ptr<Text>, float>& p){
                       p.second-=secondsPerTick;
                       return p.second<=0;}
                   ), texts.end());
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


    for(std::pair<std::unique_ptr<Text>, float> &p : texts){

        //make this text face towards the camera
        p.first->calculateModel();
        p.first->model =  p.first->model * v;

        p.first->draw(textShader);
    }
    textShader->unbind();
}
