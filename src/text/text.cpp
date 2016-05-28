#include "saiga/text/all.h"
#include "saiga/util/assert.h"


Text::Text(TextureAtlas *textureAtlas, const std::string &label, bool normalize):
    size(label.size()),capacity(label.size()),normalize(normalize),label(label),textureAtlas(textureAtlas){

    addTextToMesh(label);
    updateGLBuffer(0,true);
    calculateNormalizationMatrix();
}

void Text::calculateNormalizationMatrix()
{
    boundingBox = mesh.calculateAabb();
    normalizationMatrix = mat4();
    if(normalize){

//        float height = boundingBox.max.y - boundingBox.min.y;
        float height = 1.0f;

        vec3 offset = boundingBox.getPosition();
        normalizationMatrix[3] = vec4(-offset*1.0f/height,1);
        normalizationMatrix[0][0] = 1.0f/height;
        normalizationMatrix[1][1] = 1.0f/height;
        normalizationMatrix[2][2] = 1.0f/height;
        boundingBox.transform(normalizationMatrix);
    }else{
//        cout << "boundingBox " << boundingBox << " " << textureAtlas->getMaxCharacter() << endl;
        boundingBox.growBox(textureAtlas->getMaxCharacter());
    }

//        cout<<"text "<<label<<" "<<boundingBox<<" "<<normalize<<" "<<endl<<normalizationMatrix<<endl;

}

void Text::updateText(const std::string &l, int startIndex){
    //        cout<<"update: '"<<l<<"' Start:"<<startIndex<<" old: '"<<this->label<<"'"<<endl;
    std::string label(l);
    //checks how many leading characteres are already the same.
    //if the new text is the same as the old nothing has to be done.
    int oldStartCharacter;
    bool resize = compressText(label,startIndex,oldStartCharacter);
    label = this->label.substr(startIndex);
    if(label.size()==0){
        //no update needed
        return;
    }
//        cout<<"start "<<startIndex<<" '"<<label<<"' size "<<size<<endl;

    vec2 startOffset = startPos;

    if(startIndex>0){
        //get position of last character
        const TextureAtlas::character_info &info = textureAtlas->getCharacterInfo(oldStartCharacter);

        //x offset of first new character
        startOffset.x = this->mesh.vertices[startIndex*4].position.x - info.offset.x;
        startOffset.y = this->mesh.vertices[startIndex*4].position.y - info.offset.y;
    }

    //delete everything from startindex to end
    this->mesh.vertices.resize(startIndex*4);
    this->mesh.faces.resize(startIndex);


    //calculate new faces
    addTextToMesh(label,startOffset);

    //update gl mesh
    this->updateGLBuffer(startIndex,resize);

    calculateNormalizationMatrix();
}






void Text::render(TextShader* shader){

    shader->uploadTextureAtlas(textureAtlas->getTexture());

    shader->uploadTextParameteres(params);
    shader->uploadModel(model*normalizationMatrix);

    buffer.bind();
    buffer.draw(size*6,0); //2 triangles per character
    buffer.unbind();
}



void Text::updateGLBuffer(int start, bool resize){
    if(resize){
        mesh.createBuffers(buffer);
    }else{
        mesh.updateVerticesInBuffer(buffer,(size-start)*4,start*4);
    }
}

bool Text::compressText(std::string &str, int &start, int &oldStartCharacter){
    int newLength = str.size() + start;
    size = newLength;

    label.resize(size);

    //a resize needs to copy the complete label again
    if(newLength>capacity){
        std::copy(str.begin(),str.end(),label.begin()+start);
        capacity = newLength;
        start = 0;
//        cout<<"Increasing capacity of text '"<<label<<"' to "<<size<<endl;
        return true;
    }

    //count leading characters that are equal
    int equalChars = 0;
    for(;equalChars<(int)str.size();equalChars++){
        if(label[equalChars+start]!=str[equalChars]){
            break;
        }
    }
    start += equalChars;
    oldStartCharacter = label[start];
    std::copy(str.begin()+equalChars,str.end(),label.begin()+start);
    return false;
}


void Text::addTextToMesh(const std::string &text, vec2 offset){
//    cout << "addTextToMesh '"<<text<<"' " << offset << endl;

    vec2 position = offset;
    VertexNT verts[4];
    for(char c : text){
        //        cout<<"create text mesh "<<(int)c<<" "<<c<<endl;
        const TextureAtlas::character_info &info = textureAtlas->getCharacterInfo((int)c);

        if(c == '\n'){
            position.x = startPos.x;
            position.y -= textureAtlas->getLineSpacing();
            continue;
        }


        vec3 bufferPosition = vec3(
                    position.x + info.offset.x,
                    position.y + info.offset.y,
                    0);

//        cout << "bufferPosition '"<<c<<"' " << bufferPosition << " " << info.offset.y << endl;

        //bottom left
        verts[0] = VertexNT(bufferPosition,
                            vec3(0,0,1),
                            vec2(info.tcMin.x,info.tcMax.y));
        //bottom right
        verts[1] = VertexNT(bufferPosition+vec3(info.size.x,0,0),
                            vec3(0,0,1),
                            vec2(info.tcMax.x,info.tcMax.y));
        //top right
        verts[2] = VertexNT(bufferPosition+vec3(info.size.x,info.size.y,0),
                            vec3(0,0,1),
                            vec2(info.tcMax.x,info.tcMin.y));
        //top left
        verts[3] = VertexNT(bufferPosition+vec3(0,info.size.y,0),
                            vec3(0,0,1),
                            vec2(info.tcMin.x,info.tcMin.y));

        mesh.addQuad(verts);

        position += info.advance;
        position.x += textureAtlas->additionalCharacterSpacing;
    }
}
