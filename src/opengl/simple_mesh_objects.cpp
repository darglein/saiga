#include "saiga/opengl/simple_mesh_objects.h"
#include "saiga/geometry/triangle_mesh_generator.h"

void SphereMeshObject::createMesh(){
    Sphere s(vec3(0),1);
    auto m = TriangleMeshGenerator::createMesh(s,5);
    m->createBuffers(buffer);
}

SphereMeshObject::SphereMeshObject(const Sphere &sphere):SphereMeshObject(sphere,vec4(1)){

}

SphereMeshObject::SphereMeshObject()
{
}

SphereMeshObject::SphereMeshObject(const Sphere &sphere, const vec4 &color):SimpleMeshObject(color){
    setScale(vec3(sphere.r));
    translateGlobal(sphere.pos);
}



 //==============================================================

 void PlaneMeshObject::createMesh(){
     Plane p(vec3(0),vec3(0,1,0));
     auto m = TriangleMeshGenerator::createMesh(p);
     m->createBuffers(buffer);
 }

 PlaneMeshObject::PlaneMeshObject(const Plane &plane):PlaneMeshObject(plane,vec4(1)){}

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

 ConeMeshObject::ConeMeshObject(const Cone &cone):ConeMeshObject(cone,vec4(1)){}

 ConeMeshObject::ConeMeshObject()
 {
 }

 ConeMeshObject::ConeMeshObject(const Cone &cone, const vec4 &color):SimpleMeshObject(color){
     translateGlobal(cone.position);
 }



