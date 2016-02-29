#include "saiga/text/all.h"

Text::Text(TextGenerator *textureAtlas):Text(textureAtlas,""){

}

Text::Text(TextGenerator *textureAtlas, const std::string &label):
    textureAtlas(textureAtlas),label(label){
}

void Text::updateText123(const std::string &l, int startIndex){
    cout<<"update: "<<l<<endl;
    std::string label(l);
    //checks how many leading characteres are already the same.
    //if the new text is the same as the old nothing has to be done.
    compressText(label,startIndex);
    if(label.size()==0){
        //no update needed
        return;
    }


    //get position of last character
    TextGenerator::character_info &info = textureAtlas->characters[(int)this->label[startIndex]];
    this->updateText(label,startIndex);

    //x offset of first new character
    int start = this->mesh.vertices[startIndex*4].position.x - info.bl;
    //delete everything from startindex to end
    this->mesh.vertices.resize(startIndex*4);
    this->mesh.faces.resize(startIndex);


    //calculate new faces
    textureAtlas->createTextMesh(this->mesh,label,start);

    //update gl mesh
    this->updateGLBuffer(startIndex);
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
