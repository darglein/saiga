/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/geometry/object3d.h"
#include "saiga/core/util/encoding.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/opengl/text/textParameters.h"

#ifndef SAIGA_USE_FREETYPE
#    error Saiga was compiled without freetype.
#endif

namespace Saiga
{
class TextShader;
class TextAtlas;

class SAIGA_OPENGL_API Text : public Object3D
{
   public:
    mat4 normalizationMatrix;
    TextParameters params;
    bool visible = true;



    Text(TextAtlas* textureAtlas, const std::string& label = "", bool normalize = false);
    virtual ~Text() {}

    /**
     * Uploads the texture atlas, color, strokecolor and model matrix to the shader and draws the mesh.
     */
    void render(std::shared_ptr<TextShader> shader);

    /**
     * Updates the current text beginning at 'startIndex'.
     * This may change the size of the OpenGL buffer.
     */
    void updateText(const std::string& label, int startIndex = 0);
    std::string getText();

    /**
     * Returns the bounding box of the text mesh.
     * The mesh is always in the x-y Plane ( z=0 ).
     * If normalized is set to true the center of the mesh is vec3(0).
     */
    AABB getAabb() { return boundingBox; }
    vec3 getSize() { return boundingBox.max - boundingBox.min; }
    int getLines() { return lines; }


   private:
    // similar to a std::vector the capacity is not decreased when the size is decreased
    int size;       // current size of the label
    int capacity;   // size of the gpu buffer
    int lines = 1;  // number of '\n' + 1

    vec2 startPos = vec2(0, 0);  // bottom left corner of first character

    bool normalize;  // normalized text is centered around the origin
    //    std::string label;
    utf32string label;
    TriangleMesh<VertexNT, GLuint> mesh;
    IndexedVertexBuffer<VertexNT, GLuint> buffer;
    TextAtlas* textureAtlas;
    AABB boundingBox;

    void calculateNormalizationMatrix();
    void updateGLBuffer(int start, bool resize);
    bool compressText(utf32string& str, int& start, int& lines);

    // adds 'text' to the end of this triangle mesh. This will add 4 vertices and 4 indices per character (2 Triangles).
    void addTextToMesh(const utf32string& text, vec2 offset = vec2(0, 0));
};

}  // namespace Saiga
