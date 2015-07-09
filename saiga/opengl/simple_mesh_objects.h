#pragma once

#include "saiga/opengl/mesh_object.h"
#include "saiga/opengl/basic_shaders.h"

class Sphere;
class Cone;
class Plane;


template<class T>
class SimpleMeshObject : public MeshObject<T,VertexNT,GLuint,MVPColorShader>
{
    vec4 color;
public:
    SimpleMeshObject():color(1){}
    SimpleMeshObject(const vec4 &color):color(color){}
    void bindUniforms();
};

template<class T>
void SimpleMeshObject<T>::bindUniforms(){
    mat4 model;
    this->getModelMatrix(model);
    this->shader->uploadModel(model);
    this->shader->uploadColor(this->color);
}


class SAIGA_GLOBAL SphereMeshObject :  public SimpleMeshObject<SphereMeshObject>
{
public:
    static void createMesh();
    SphereMeshObject();
    SphereMeshObject(const Sphere &sphere);
    SphereMeshObject(const Sphere &sphere, const vec4 &color);
};

class SAIGA_GLOBAL PlaneMeshObject :  public SimpleMeshObject<PlaneMeshObject>
{
public:
    static void createMesh();
    PlaneMeshObject();
    PlaneMeshObject(const Plane &plane);
    PlaneMeshObject(const Plane &plane, const vec4 &color);
};

class SAIGA_GLOBAL ConeMeshObject :  public SimpleMeshObject<ConeMeshObject>
{
public:
    static void createMesh();
    ConeMeshObject();
    ConeMeshObject(const Cone &cone);
    ConeMeshObject(const Cone &cone, const vec4 &color);
};

