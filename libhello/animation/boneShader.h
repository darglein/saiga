#pragma once

#include <libhello/opengl/vertexBuffer.h>
#include <libhello/opengl/shader.h>
#include <libhello/geometry/triangle_mesh.h>
#include <libhello/rendering/object3d.h>
#include <libhello/opengl/mesh.h>
#include <libhello/opengl/mesh_object.h>
#include <functional>




class BoneShader : public MVPShader{
public:
    GLuint location_boneMatrices;
    BoneShader(const string &multi_file) : MVPShader(multi_file){}
    virtual void checkUniforms();

    void uploadBoneMatrices(mat4* matrices, int count);
};



