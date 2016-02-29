#include "saiga/text/all.h"
#include "saiga/util/assert.h"


Text::Text(TextGenerator *textureAtlas, int size, bool normalize):
    Text(textureAtlas,std::string(size,'A'),normalize)
{
}

Text::Text(TextGenerator *textureAtlas, const std::string &label, bool normalize):
    textureAtlas(textureAtlas),label(label),size(label.size()){
    this->texture = textureAtlas->textureAtlas;

    addTextToMesh(label);
    if(normalize){
        mesh.boundingBox.growBox(textureAtlas->maxCharacter);
        aabb bb = mesh.getAabb();
        vec3 offset = bb.getPosition();
        mat4 t;
        t[3] = vec4(-offset,0);
        mesh.transform(t);
    }
    mesh.createBuffers(this->buffer);
}



void Text::updateText123(const std::string &l, int startIndex){
//    cout<<"update: '"<<l<<"' Start:"<<startIndex<<" old: '"<<this->label<<"'"<<endl;
    std::string label(l);
    //checks how many leading characteres are already the same.
    //if the new text is the same as the old nothing has to be done.
    compressText(label,startIndex);
    label = this->label.substr(startIndex);
    if(label.size()==0){
        //no update needed
        return;
    }

    if(startIndex==this->label.size()){
        return;
    }


    //get position of last character
    TextGenerator::character_info &info = textureAtlas->characters[(int)this->label[startIndex]];

    //x offset of first new character
    int startX = this->mesh.vertices[startIndex*4].position.x - info.bl;
    //delete everything from startindex to end
    int verticesBefore = this->mesh.vertices.size();
    this->mesh.vertices.resize(startIndex*4);
    this->mesh.faces.resize(startIndex);


    //calculate new faces

//    cout<<"label: '"<<label<<"' '"<<this->label<<"'"<<endl;
//    textureAtlas->createTextMesh(this->mesh,label,startX);
    addTextToMesh(label,startX);

    //update gl mesh
    this->updateGLBuffer(startIndex);

    assert(verticesBefore==this->mesh.vertices.size());
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

    //if str is longer than size, cut it off
    int s = min(static_cast<int>(str.size()),size-start);
    str.resize(s);

    //count leading characters that are equal
    int equalChars = 0;
    for(;equalChars<str.size();equalChars++){
        if(label[equalChars+start]!=str[equalChars]){
            break;
        }
    }
    start += equalChars;

    std::copy(str.begin()+equalChars,str.end(),label.begin()+start);

}


void Text::addTextToMesh(const std::string &text, int startX, int startY){

    int x=startX,y=startY;
    VertexNT verts[4];
    for(char c : text){
//        cout<<"create text mesh "<<(int)c<<" "<<c<<endl;
        TextGenerator::character_info &info = textureAtlas->characters[(int)c];

        vec3 offset = vec3(x+info.bl,y+info.bt-info.bh,0);


        //bottom left
        verts[0] = VertexNT(offset,
                            vec3(0,0,1),
                            vec2(info.tcMin.x,info.tcMax.y));
        //bottom right
        verts[1] = VertexNT(offset+vec3(info.bw,0,0),
                            vec3(0,0,1),
                            vec2(info.tcMax.x,info.tcMax.y));
        //top right
        verts[2] = VertexNT(offset+vec3(info.bw,info.bh,0),
                            vec3(0,0,1),
                            vec2(info.tcMax.x,info.tcMin.y));
        //top left
        verts[3] = VertexNT(offset+vec3(0,info.bh,0),
                            vec3(0,0,1),
                            vec2(info.tcMin.x,info.tcMin.y));

        x+=info.ax;
        y+=info.ay;
        mesh.addQuad(verts);
    }
}
