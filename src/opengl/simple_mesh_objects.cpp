#include "libhello/opengl/simple_mesh_objects.h"


void SphereMeshObject::createMesh(){
    Sphere s(vec3(0),1);
    auto m = TriangleMeshGenerator::createMesh(s,5);
    m->createBuffers(buffer);
}

SphereMeshObject::SphereMeshObject()
{
}

SphereMeshObject::SphereMeshObject(const Sphere &sphere, const vec4 &color):SimpleMeshObject(color){
    scale(vec3(sphere.r));
    translateGlobal(sphere.pos);
}



 //==============================================================

 void PlaneMeshObject::createMesh(){
     Plane p(vec3(0),vec3(0,1,0));
     auto m = TriangleMeshGenerator::createMesh(p);
     m->createBuffers(buffer);
 }

 PlaneMeshObject::PlaneMeshObject()
 {
 }

 PlaneMeshObject::PlaneMeshObject(const Plane &plane, const vec4 &color):SimpleMeshObject(color){
     translateGlobal(plane.point);
 }

 //==============================================================

 void ConeMeshObject::createMesh(){
     Cone c(vec3(0),vec3(0,1,0),30.0f,10.0f);
     auto m = TriangleMeshGenerator::createMesh(c,5);
     m->createBuffers(buffer);
 }

 ConeMeshObject::ConeMeshObject()
 {
 }

 ConeMeshObject::ConeMeshObject(const Cone &cone, const vec4 &color):SimpleMeshObject(color){
     translateGlobal(cone.position);
 }



