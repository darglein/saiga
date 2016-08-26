#pragma once

#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/rendering/object3d.h"

#define MO_TEMPLATE_TYPES typename T,typename vertex_t,typename index_t,typename shader_t
#define MO_TEMPLATES T,vertex_t,index_t,shader_t


//using: Curiously recurring template pattern
//http://en.wikipedia.org/wiki/Curiously_recurring_template_pattern

class SAIGA_GLOBAL MeshObjectParent:  public Object3D{
public:
    virtual void draw() = 0;
    virtual void drawNoShaderBind() = 0;
};



template<MO_TEMPLATE_TYPES>
class SAIGA_TEMPLATE MeshObject :  public MeshObjectParent{
public:
    static IndexedVertexBuffer<vertex_t,index_t> buffer;
    static shader_t* shader;

    MeshObject(){}


public:
    void draw();
    void drawNoShaderBind();
    void drawRaw();
};

template<MO_TEMPLATE_TYPES>
IndexedVertexBuffer<vertex_t,index_t> MeshObject<MO_TEMPLATES>::buffer;
template<MO_TEMPLATE_TYPES>
shader_t*  MeshObject<MO_TEMPLATES>::shader = nullptr;

template<MO_TEMPLATE_TYPES>
void MeshObject<MO_TEMPLATES>::draw(){
    shader->bind();
    static_cast<T*>(this)->bindUniforms();
    buffer.bindAndDraw();
    shader->unbind();
}

template<MO_TEMPLATE_TYPES>
void MeshObject<MO_TEMPLATES>::drawNoShaderBind(){
    static_cast<T*>(this)->bindUniforms();
    buffer.bindAndDraw();
}

template<MO_TEMPLATE_TYPES>
void MeshObject<MO_TEMPLATES>::drawRaw(){
    buffer.bindAndDraw();
}

