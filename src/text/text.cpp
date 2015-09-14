#include "saiga/text/text.h"

#include "saiga/text/textShader.h"

Text::Text(const std::string &label):label(label){
}

void Text::updateText(const std::string &label){
    cout<<"update: "<<label<<endl;
}

void Text::draw(TextShader* shader){

    shader->upload(texture,color,strokeColor);
    shader->uploadModel(model);

    buffer.bindAndDraw();
}


