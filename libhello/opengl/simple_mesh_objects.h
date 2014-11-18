#pragma once

#include "libhello/opengl/mesh_object.h"
#include "libhello/opengl/basic_shaders.h"
#include "libhello/geometry/triangle_mesh_generator.h"

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
    this->shader->uploadModel(this->model);
    this->shader->uploadColor(this->color);
}


class SphereMeshObject :  public SimpleMeshObject<SphereMeshObject>
{
public:
    static void createMesh();
    SphereMeshObject();
    SphereMeshObject(const Sphere &sphere):SphereMeshObject(sphere,vec4(1)){}
    SphereMeshObject(const Sphere &sphere, const vec4 &color);
};

class PlaneMeshObject :  public SimpleMeshObject<PlaneMeshObject>
{
public:
    static void createMesh();
    PlaneMeshObject();
    PlaneMeshObject(const Plane &plane):PlaneMeshObject(plane,vec4(1)){}
    PlaneMeshObject(const Plane &plane, const vec4 &color);
};

class ConeMeshObject :  public SimpleMeshObject<ConeMeshObject>
{
public:
    static void createMesh();
    ConeMeshObject();
    ConeMeshObject(const Cone &cone):ConeMeshObject(cone,vec4(1)){}
    ConeMeshObject(const Cone &cone, const vec4 &color);
};

