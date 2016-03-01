#pragma once

#include "saiga/rendering/object3d.h"
#include "saiga/geometry/triangle_mesh.h"

#include <iostream>

class TextShader;
class TextureAtlas;

class SAIGA_GLOBAL Text : public Object3D{
public:
    bool visible = true;
    vec4 color=vec4(1), strokeColor=vec4(0,0,0,1);

    Text(TextureAtlas* textureAtlas, const std::string &label="", bool normalize=false);
    virtual ~Text(){}

    /**
     * Uploads the texture atlas, color, strokecolor and model matrix to the shader and draws the mesh.
     */
    void render(TextShader* shader);

    /**
     * Updates the current text beginning at 'startIndex'.
     * This may change the size of the OpenGL buffer.
     */
    void updateText(const std::string &label, int startIndex=0);
    std::string getText(){ return label;}

    /**
     * Returns the bounding box of the text mesh.
     * The mesh is always in the x-y Plane ( z=0 ).
     * If normalized is set to true the center of the mesh is vec3(0).
     */
    aabb getAabb(){return boundingBox;}
    vec3 getSize(){ return boundingBox.max-boundingBox.min;}

private:
    //similar to a std::vector the capacity is not decreased when the size is decreased
    int size; //current size of the label
    int capacity; //size of the gpu buffer

    bool normalize; //normalized text is centered around the origin
    std::string label;
    TriangleMesh<VertexNT,GLuint> mesh;
    IndexedVertexBuffer<VertexNT,GLuint> buffer;
    TextureAtlas* textureAtlas;
    mat4 normalizationMatrix;
    aabb boundingBox;

    void calculateNormalizationMatrix();
    void updateGLBuffer(int start, bool resize);
    bool compressText(std::string &str, int &start);

    //adds 'text' to the end of this triangle mesh. This will add 4 vertices and 4 indices per character (2 Triangles).
    void addTextToMesh(const std::string &text, int startX=0, int startY=0);
};
