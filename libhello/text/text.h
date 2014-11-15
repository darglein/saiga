#pragma once
#include "libhello/opengl/vertexBuffer.h"
#include "libhello/geometry/triangle_mesh.h"
#include "libhello/opengl/mesh.h"
#include <iostream>



class Text : public Object3D{
public:
    vec3 color;
    string label;
    TriangleMesh<VertexNT,GLuint> mesh;
    IndexedVertexBuffer<VertexNT,GLuint> buffer;
    Texture* texture;
    Text(){}
    Text(const string &label);
    void draw(TextShader* shader);
    void transform(const mat4 &trafo){mesh.transform(trafo);}
    void updateText(const string &label);

    vec3 getSize(){ return mesh.getAabb().max-mesh.getAabb().min;}


};
