/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/camera/camera.h"
#include "saiga/core/geometry/aabb.h"
#include "saiga/core/geometry/triangle_mesh.h"
#include "saiga/core/model/Models.h"
#include "saiga/opengl/animation/animation.h"
#include "saiga/opengl/animation/boneVertex.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/opengl/shader/basic_shaders.h"

#include <memory>

namespace Saiga
{
class SAIGA_OPENGL_API Asset
{
   public:
    virtual ~Asset() {}
    virtual void render(Camera* cam, const mat4& model)          = 0;
    virtual void renderForward(Camera* cam, const mat4& model)   = 0;
    virtual void renderDepth(Camera* cam, const mat4& model)     = 0;
    virtual void renderWireframe(Camera* cam, const mat4& model) = 0;
    virtual void renderRaw()                                     = 0;
};


template <typename ModelType>
class SAIGA_TEMPLATE BasicAsset : public Asset, public ModelType
{
   public:
    using VertexType = typename ModelType::VertexType;
    using IndexType  = typename ModelType::IndexType;

    std::shared_ptr<MVPShader> shader;
    std::shared_ptr<MVPShader> forwardShader;
    std::shared_ptr<MVPShader> depthshader;
    std::shared_ptr<MVPShader> wireframeshader;

    IndexedVertexBuffer<VertexType, IndexType> buffer;

    //    ModelType model;

    /**
     * Use these for simple inefficient rendering.
     * Every call binds and unbinds the shader and uploads the camera matrices again.
     */

    virtual ~BasicAsset() {}
    virtual void render(Camera* cam, const mat4& model) override;
    virtual void renderForward(Camera* cam, const mat4& model) override;
    virtual void renderDepth(Camera* cam, const mat4& model) override;
    virtual void renderWireframe(Camera* cam, const mat4& model) override;

    /**
     * Renders the mesh.
     * This maps to a single glDraw call and nothing else, so the shader
     * has to be setup before this renderRaw is called.
     */
    virtual void renderRaw() override;


    void create(std::shared_ptr<MVPShader> shader, std::shared_ptr<MVPShader> forwardShader,
                std::shared_ptr<MVPShader> depthshader, std::shared_ptr<MVPShader> wireframeshader,
                bool normalizePosition = false, bool ZUPtoYUP = false);
};

template <typename ModelType>
void BasicAsset<ModelType>::render(Camera* cam, const mat4& model)
{
    (void)cam;
    shader->bind();
    //    shader->uploadAll(cam,model);
    shader->uploadModel(model);

    //    glEnable(GL_POLYGON_OFFSET_FILL);
    //    glPolygonOffset(1.0f,1.0f);

    buffer.bindAndDraw();

    //    glDisable(GL_POLYGON_OFFSET_FILL);

    shader->unbind();
}

template <typename ModelType>
void BasicAsset<ModelType>::renderForward(Camera* cam, const mat4& model)
{
    (void)cam;
    forwardShader->bind();
    //    shader->uploadAll(cam,model);
    forwardShader->uploadModel(model);

    //    glEnable(GL_POLYGON_OFFSET_FILL);
    //    glPolygonOffset(1.0f,1.0f);

    buffer.bindAndDraw();

    //    glDisable(GL_POLYGON_OFFSET_FILL);

    forwardShader->unbind();
}

template <typename ModelType>
void BasicAsset<ModelType>::renderDepth(Camera* cam, const mat4& model)
{
    (void)cam;
    depthshader->bind();
    depthshader->uploadModel(model);
    buffer.bindAndDraw();
    depthshader->unbind();
}

template <typename ModelType>
void BasicAsset<ModelType>::renderWireframe(Camera* cam, const mat4& model)
{
    (void)cam;
    wireframeshader->bind();
    wireframeshader->uploadModel(model);

    //    glEnable(GL_POLYGON_OFFSET_LINE);

    // negative values shifts the wireframe towards the camera,
    // but a non zero factors does strange things for lines and increases
    // the depth on lines with high slope towards the camera by too much.
    // a visually better solution is to shift the triangles back a bit glPolygonOffset(1,1);
    // and draw the wireframe without polygon offset.
    //    glPolygonOffset(0.0f,-500.0f);

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    buffer.bindAndDraw();
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    //    glDisable(GL_POLYGON_OFFSET_LINE);

    wireframeshader->unbind();
}

template <typename ModelType>
void BasicAsset<ModelType>::renderRaw()
{
    buffer.bindAndDraw();
}

template <typename ModelType>
void BasicAsset<ModelType>::create(std::shared_ptr<MVPShader> _shader, std::shared_ptr<MVPShader> _forwardShader,
                                   std::shared_ptr<MVPShader> _depthshader, std::shared_ptr<MVPShader> _wireframeshader,
                                   bool normalizePosition, bool ZUPtoYUP)
{
    this->shader          = _shader;
    this->forwardShader   = _forwardShader;
    this->depthshader     = _depthshader;
    this->wireframeshader = _wireframeshader;
    //    this->model.boundingBox = model.mesh.aabb();

    if (ZUPtoYUP)
    {
        //        this->model.ZUPtoYUP();
    }

    if (normalizePosition)
    {
        this->normalizePosition();
    }
    buffer.fromMesh(*this);
}

}  // namespace Saiga
