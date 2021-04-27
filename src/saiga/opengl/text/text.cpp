/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/util/assert.h"
#include "saiga/core/util/encoding.h"
#include "saiga/opengl/text/all.h"

namespace Saiga
{
Text::Text(TextAtlas* textureAtlas, const std::string& label, bool normalize)
    : normalize(normalize), textureAtlas(textureAtlas)
{
    this->label = Encoding::UTF8toUTF32(label);
    //    this->label = label;
    size     = this->label.size();
    capacity = this->label.size();

    addTextToMesh(this->label);
    updateGLBuffer(0, true);
    calculateNormalizationMatrix();
}

void Text::calculateNormalizationMatrix()
{
    boundingBox         = mesh.aabb();
    normalizationMatrix = mat4::Identity();
    if (normalize)
    {
        //        float height = boundingBox.max[1] - boundingBox.min[1];
        float height = 1.0f;

        vec3 offset                   = boundingBox.getPosition();
        normalizationMatrix.col(0)[0] = 1.0f / height;
        normalizationMatrix.col(1)[1] = 1.0f / height;
        normalizationMatrix.col(2)[2] = 1.0f / height;
        normalizationMatrix.col(3)    = make_vec4(-offset * 1.0f / height, 1);
        boundingBox.transform(normalizationMatrix);
    }
    else
    {
        //        std::cout << "boundingBox " << boundingBox << " " << textureAtlas->getMaxCharacter() << std::endl;

        // grow every line by max character
        AABB maxCharacter = textureAtlas->getMaxCharacter();
        for (int i = 0; i < lines; ++i)
        {
            boundingBox.growBox(maxCharacter);
            maxCharacter.translate(vec3(0, -textureAtlas->getLineSpacing(), 0));
        }
    }

    //        std::cout<<"text "<<label<<" "<<boundingBox<<" "<<normalize<<" "<<endl<<normalizationMatrix<<endl;
}

void Text::updateText(const std::string& l, int startIndex)
{
    //    std::cout<<"Text::updateText: '"<<l<<"' Start:"<<startIndex<<" old: '"<<this->label<<"'"<<endl;
    //    std::string label(l);
    utf32string label = Encoding::UTF8toUTF32(l);
    // checks how many leading characteres are already the same.
    // if the new text is the same as the old nothing has to be done.
    lines       = 1;
    bool resize = compressText(label, startIndex, lines);
    label       = utf32string(this->label.begin() + startIndex, this->label.end());

    if (label.size() == 0)
    {
        // no update needed
        return;
    }
    //        std::cout<<"start "<<startIndex<<" '"<<label<<"' size "<<size << " resize=" <<resize << "
    //        oldStartCharacter="<<(char)oldStartCharacter<<endl;

    vec2 startOffset = startPos;

    if (startIndex > 0)
    {
        // get position of last character before startindex
        int lastCharPos       = startIndex - 1;
        int oldStartCharacter = this->label[lastCharPos];

        const TextAtlas::character_info& info = textureAtlas->getCharacterInfo(oldStartCharacter);

        startOffset[0] = this->mesh.vertices[lastCharPos * 4].position[0];
        startOffset[1] = this->mesh.vertices[lastCharPos * 4].position[1];

        // calculate start position of next character
        startOffset -= info.offset;
        startOffset += info.advance;
        startOffset[0] += textureAtlas->additionalCharacterSpacing;
    }

    // delete everything from startindex to end
    this->mesh.vertices.resize(startIndex * 4);
    this->mesh.faces.resize(startIndex);


    // calculate new faces
    addTextToMesh(label, startOffset);

    // update gl mesh
    this->updateGLBuffer(startIndex, resize);

    calculateNormalizationMatrix();
}

std::string Text::getText()
{
    return Encoding::UTF32toUTF8(label);
    //    return label;
}



void Text::render(std::shared_ptr<TextShader> shader)
{
    shader->uploadTextureAtlas(textureAtlas->getTexture());

    shader->uploadTextParameteres(params);
    shader->uploadModel(model * normalizationMatrix);

    buffer.bind();
    buffer.draw(size * 6, 0);  // 2 triangles per character
    buffer.unbind();
}



void Text::updateGLBuffer(int start, bool resize)
{
    if (resize)
    {
        //        mesh.createBuffers(buffer,GL_DYNAMIC_DRAW);
        buffer.fromMesh(mesh, GL_DYNAMIC_DRAW);
    }
    else
    {
        //        std::cout<<"Text::updateGLBuffer "<<"start="<<start << " " << mesh.vertices.size() << " " <<
        //        (size-start)*4
        //        << std::endl; mesh.updateVerticesInBuffer(buffer,(size-start)*4,start*4);
        buffer.updateFromMesh(mesh, (size - start) * 4, start * 4);
    }
}

bool Text::compressText(utf32string& str, int& start, int& lines)
{
    int newLength = str.size() + start;
    size          = newLength;

    label.resize(size);

    // a resize needs to copy the complete label again
    if (newLength > capacity)
    {
        std::copy(str.begin(), str.end(), label.begin() + start);
        capacity = newLength;
        start    = 0;
        //        std::cout<<"Increasing capacity of text '"<<label<<"' to "<<size<<endl;
        return true;
    }

    // count leading characters that are equal
    // count new line characters
    int equalChars = 0;
    for (; equalChars < (int)str.size(); equalChars++)
    {
        if (label[equalChars + start] != str[equalChars])
        {
            break;
        }
        if (str[equalChars] == '\n')
        {
            lines++;
        }
    }
    start += equalChars;
    std::copy(str.begin() + equalChars, str.end(), label.begin() + start);
    return false;
}


void Text::addTextToMesh(const utf32string& text, vec2 offset)
{
    //    std::cout << "addTextToMesh '"<<text<<"' " << offset << std::endl;

    //    std::vector<uint32_t> utf32string = Encoding::UTF8toUTF32(text);

    //    std::cout<<"convert back "<<Encoding::UTF32toUTF8(utf32string)<<endl;
    vec2 position = offset;
    VertexNT verts[4];
    for (uint32_t c : text)
    {
        //        std::cout<<"create text mesh "<<std::dec<<(int)c<<" "<<std::hex<<(int)c<<" "<<c<<endl;
        const TextAtlas::character_info& info = textureAtlas->getCharacterInfo((int)c);

        if (c == '\n')
        {
            position[0] = startPos[0];
            position[1] -= textureAtlas->getLineSpacing();
            lines++;
            // emit a degenerated quad
            // TODO: maybe remove this and count 'actual' characters
        }


        vec3 bufferPosition = vec3(position[0] + info.offset[0], position[1] + info.offset[1], 0);

        //        std::cout << "bufferPosition '"<<(char)c<<"' " << bufferPosition << " " << info.offset[1] <<
        //        std::endl;

        // bottom left
        verts[0] = VertexNT(bufferPosition, vec3(0, 0, 1), vec2(info.tcMin[0], info.tcMax[1]));
        // bottom right
        verts[1] =
            VertexNT(bufferPosition + vec3(info.size[0], 0, 0), vec3(0, 0, 1), vec2(info.tcMax[0], info.tcMax[1]));
        // top right
        verts[2] = VertexNT(bufferPosition + vec3(info.size[0], info.size[1], 0), vec3(0, 0, 1),
                            vec2(info.tcMax[0], info.tcMin[1]));
        // top left
        verts[3] =
            VertexNT(bufferPosition + vec3(0, info.size[1], 0), vec3(0, 0, 1), vec2(info.tcMin[0], info.tcMin[1]));

        mesh.addQuad(verts);

        position += info.advance;
        position[0] += textureAtlas->additionalCharacterSpacing;
    }
}

}  // namespace Saiga
