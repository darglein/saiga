#include "text/text.h"

Text::Text(const string &label):label(label){





}

void Text::updateText(const string &label){
    cout<<"UPdate: "<<label<<endl;
}

void Text::draw(TextShader* shader){

    shader->upload(texture,color);
    shader->uploadModel(model);

    buffer.bindAndDraw();
}


