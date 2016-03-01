#include "saiga/text/TextOverlay3D.h"
#include "saiga/text/textShader.h"
#include "saiga/text/text.h"
#include "saiga/camera/camera.h"
#include "saiga/opengl/shader/shaderLoader.h"

#include <algorithm>

const float TextOverlay3D::INFINITE_DURATION = -1.f;

bool TextOverlay3D::TextContainer::update(float delta)
{
    text->position += velocity*delta;
    if (duration == INFINITE_DURATION)
        return false;
    duration-=delta;
    return duration<=0;
}

float TextOverlay3D::TextContainer::getFade()
{

    float a = 1.0f;
    if(fade){

        if(duration<fadeStart)
            a = duration / (maxDuration-fadeStart) ;
    }
    return a;
}


TextOverlay3D::TextOverlay3D(){

}

void TextOverlay3D::render(Camera *cam){
    renderText(cam);
}

void TextOverlay3D::addText(std::shared_ptr<Text> text, float duration, bool orientToCamera){
    texts.push_back(TextContainer(std::move(text), duration, orientToCamera));
}

void TextOverlay3D::addText(const TextOverlay3D::TextContainer &tc)
{
    texts.push_back(tc);
}

void TextOverlay3D::update(float secondsPerTick)
{

    auto func = [secondsPerTick](TextContainer& p){
        return p.update(secondsPerTick);

    };

    texts.erase(std::remove_if(texts.begin(), texts.end(),func ), texts.end());
}



void TextOverlay3D::renderText(Camera *cam){

    textShader->bind();

    textShader->uploadProj(cam->proj);
    textShader->uploadView(cam->view);

    mat4 v = cam->model;
    v[3] = vec4(0,0,0,1);


    for(TextContainer &p : texts){
        textShader->uploadFade(p.getFade());
        if (p.orientToCamera){
            //make this text face towards the camera
            p.text->calculateModel();
            p.text->model =  p.text->model * v;
        }


        p.text->render(textShader);
    }
    textShader->unbind();
}


void TextOverlay3D::loadShader()
{
    if(textShader!=nullptr)
        return;
    textShader = ShaderLoader::instance()->load<TextShaderFade>("deferred_text3D.glsl");

}
