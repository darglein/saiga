#pragma once

#include "libhello/rendering/object3d.h"
#include "libhello/opengl/vertexBuffer.h"
#include "libhello/geometry/triangle_mesh.h"

#include <iostream>


//#include "libhello/opengl/mesh.h"
class TextShader;
class basic_Texture_2D;

class SAIGA_GLOBAL Text : public Object3D{
public:
    bool visible = true;
    vec4 color=vec4(1), strokeColor=vec4(0,0,0,1);
    std::string label;
    TriangleMesh<VertexNT,GLuint> mesh;
    IndexedVertexBuffer<VertexNT,GLuint> buffer;
    basic_Texture_2D* texture;
    Text(){}
    Text(const std::string &label);
    virtual ~Text(){}
    void draw(TextShader* shader);
    void transform(const mat4 &trafo){mesh.transform(trafo);}
    void updateText(const std::string &label);

    vec3 getSize(){ return mesh.getAabb().max-mesh.getAabb().min;}


};
