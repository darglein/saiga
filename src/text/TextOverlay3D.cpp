#include "text/TextOverlay3D.h"
#include "libhello/opengl/shader.h"
#include "libhello/text/text.h"
#include "libhello/camera/camera.h"
#include <algorithm>

const float TextOverlay3D::INFINITE_DURATION = -1.f;

TextOverlay3D::TextOverlay3D(){

}

void TextOverlay3D::render(Camera *cam){
    renderText(cam);
}

void TextOverlay3D::addText(std::shared_ptr<Text> text, float duration, bool orientToCamera){
    texts.push_back(TextContainer(std::move(text), duration, orientToCamera));
}

void TextOverlay3D::update(float secondsPerTick)
{
    texts.erase(std::remove_if(texts.begin(), texts.end(),
                   [secondsPerTick](TextContainer& p){
                    if (p.duration == INFINITE_DURATION)return false;
                       p.duration-=secondsPerTick;
                       return p.duration<=0;}
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


    for(TextContainer &p : texts){

        if (p.orientToCamera){
            //make this text face towards the camera
            p.text->calculateModel();
            p.text->model =  p.text->model * v;
        }


        p.text->draw(textShader);
    }
    textShader->unbind();
}
