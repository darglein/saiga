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



void Text::updateGLBuffer(int start){
    mesh.updateVerticesInBuffer(buffer,(size-start)*4,start*4);
}

void Text::compressText(std::string &str, int &start){

    int s = min(static_cast<int>(str.size()),size-start);
    str.resize(s);

    int newSize = 0;
    bool found = false;
    for(unsigned int i=0;i<str.size();i++){
        if(found || label[i+start]!=str[i]){
            str[newSize++] = str[i];
            found = true;
        }else{

        }
    }

    start = start + (str.size()-newSize);
    str.resize(newSize);
}

char Text::updateText(std::string &str, int start){

    char c = label[start];
    int s = str.size();
//    cout<<s<<" "<<str.size()<<endl;
    str.resize(size-start);



    for(int i=start;i<size;i++){
        if(i<start+s){

            label[i] = str[i-start];
        }else{
            str[i-start] = label[i];
        }
    }
    return c;
}
