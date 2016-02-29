#pragma once

#include "saiga/rendering/object3d.h"
#include "saiga/geometry/triangle_mesh.h"

#include <iostream>


//#include "saiga/opengl/mesh.h"
class TextShader;
class basic_Texture_2D;
class TextGenerator;

class SAIGA_GLOBAL Text : public Object3D{
public:
    int size; //dynamic text has fixed size
    bool visible = true;
    vec4 color=vec4(1), strokeColor=vec4(0,0,0,1);
    std::string label;
    TriangleMesh<VertexNT,GLuint> mesh;
    IndexedVertexBuffer<VertexNT,GLuint> buffer;
    basic_Texture_2D* texture;
    TextGenerator* textureAtlas;

    Text(TextGenerator* textureAtlas);
    Text(TextGenerator* textureAtlas, const std::string &label);
    virtual ~Text(){}

    void draw(TextShader* shader);
    void transform(const mat4 &trafo){mesh.transform(trafo);}

    void updateText123(const std::string &label, int startIndex);

    vec3 getSize(){ return mesh.getAabb().max-mesh.getAabb().min;}

private:
    void updateGLBuffer(int start);
    void compressText(std::string &str, int &start);

public:
    //adds 'text' to the end of this triangle mesh. This will add 4 vertices and 4 indices per character (2 Triangles).
    void addTextToMesh(const std::string &text, int startX=0, int startY=0);
};
