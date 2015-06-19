#pragma once

#include <vector>
#include "libhello/opengl/basic_shaders.h"
#include "libhello/opengl/mesh_object.h"



//class MaterialMesh : public MeshObject<MaterialMesh,VertexNT,GLuint,MaterialShader>{
//public:
//    std::string name;
//    std::vector<TriangleGroup> triangleGroups;
//    MaterialMesh(){}
//    virtual ~MaterialMesh(){}

//    void addTriangleGroup(const TriangleGroup &tg){
//        triangleGroups.push_back(tg);
//    }

//    void bindUniforms(){};
//    void draw(const mat4 &model,const Camera &cam);
//};

class FBMesh : public MeshObject<FBMesh,VertexNT,GLuint,FBShader>{
public:
    Framebuffer* framebuffer;
    FBMesh();
    void bindUniforms();
    void createQuadMesh();
//    void draw(const Camera &cam);
};

