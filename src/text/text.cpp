#include "text/text.h"
#include "libhello/opengl/basic_shaders.h"
Text::Text(const std::string &label):label(label){





}

void Text::updateText(const std::string &label){
    cout<<"UPdate: "<<label<<endl;
}

void Text::draw(TextShader* shader){

    shader->upload(texture,color,strokeColor);
    shader->uploadModel(model);

    buffer.bindAndDraw();
}


