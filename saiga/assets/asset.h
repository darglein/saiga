#pragma once

#include <saiga/opengl/indexedVertexBuffer.h>
#include <saiga/opengl/shader/basic_shaders.h>
#include <saiga/geometry/triangle_mesh.h>
#include <saiga/geometry/aabb.h>
#include <saiga/animation/boneVertex.h>
#include <saiga/animation/animation.h>

#include <saiga/camera/camera.h>


class SAIGA_GLOBAL Asset{
public:
    virtual void render(Camera *cam, const mat4 &model) = 0;
    virtual void renderDepth(Camera *cam, const mat4 &model) = 0;
    virtual void renderRaw() = 0;
};


template<typename vertex_t, typename index_t>
class SAIGA_GLOBAL BasicAsset : public Asset{
public:
    std::string name;
    aabb boundingBox;

    MVPShader* shader  = nullptr;
    MVPShader* depthshader  = nullptr;

    TriangleMesh<vertex_t,index_t> mesh;
    IndexedVertexBuffer<vertex_t,index_t> buffer;

    /**
     * Use these for simple inefficient rendering.
     * Every call binds and unbinds the shader and uploads the camera matrices again.
     */

    void render(Camera *cam, const mat4 &model);
    void renderDepth(Camera *cam, const mat4 &model);

    /**
     * Renders the mesh.
     * This maps to a single glDraw call and nothing else, so the shader
     * has to be setup before this renderRaw is called.
     */
    void renderRaw();

};

template<typename vertex_t, typename index_t>
void BasicAsset<vertex_t,index_t>::render(Camera *cam, const mat4 &model)
{
    shader->bind();
    shader->uploadAll(model,cam->view,cam->proj);
    buffer.bindAndDraw();
    shader->unbind();
}

template<typename vertex_t, typename index_t>
void BasicAsset<vertex_t,index_t>::renderDepth(Camera *cam, const mat4 &model)
{
    depthshader->bind();
    depthshader->uploadAll(model,cam->view,cam->proj);
    buffer.bindAndDraw();
    depthshader->unbind();
}

template<typename vertex_t, typename index_t>
void BasicAsset<vertex_t,index_t>::renderRaw()
{
    buffer.bindAndDraw();
}
