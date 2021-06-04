/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/camera/camera.h"
#include "saiga/core/geometry/aabb.h"
#include "saiga/core/geometry/triangle_mesh.h"
#include "saiga/core/model/all.h"
#include "saiga/core/model/animation.h"
#include "saiga/opengl/UnifiedMeshBuffer.h"
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


template <typename ShaderType>
class SAIGA_TEMPLATE BasicAsset : public Asset
{
   public:

    std::shared_ptr<ShaderType> deferredShader;
    std::shared_ptr<ShaderType> forwardShader;
    std::shared_ptr<ShaderType> depthshader;
    std::shared_ptr<ShaderType> wireframeshader;

    std::shared_ptr<UnifiedMeshBuffer> unified_buffer;
    std::vector<UnifiedMaterialGroup> groups;

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


    virtual void loadDefaultShaders() = 0;

    void setShader(std::shared_ptr<ShaderType> deferredShader, std::shared_ptr<ShaderType> forwardShader,
                   std::shared_ptr<ShaderType> depthshader, std::shared_ptr<ShaderType> wireframeshader);

};

template <typename ShaderType>
void BasicAsset< ShaderType>::render(Camera* cam, const mat4& model)
{
    (void)cam;
    SAIGA_ASSERT(deferredShader);
    if(deferredShader->bind())
    {
        //    shader->uploadAll(cam,model);
        deferredShader->uploadModel(model);

        //    glEnable(GL_POLYGON_OFFSET_FILL);
        //    glPolygonOffset(1.0f,1.0f);

        if (unified_buffer)
        {
            unified_buffer->Bind();
            unified_buffer->Draw();
            unified_buffer->Unbind();
        }

        //    glDisable(GL_POLYGON_OFFSET_FILL);

        deferredShader->unbind();
    }
}

template < typename ShaderType>
void BasicAsset< ShaderType>::renderForward(Camera* cam, const mat4& model)
{
    (void)cam;
    if(forwardShader->bind())
    {
        //    shader->uploadAll(cam,model);
        forwardShader->uploadModel(model);

        //    glEnable(GL_POLYGON_OFFSET_FILL);
        //    glPolygonOffset(1.0f,1.0f);

        if (unified_buffer)
        {
            unified_buffer->Bind();
            unified_buffer->Draw();
            unified_buffer->Unbind();
        }

        //    glDisable(GL_POLYGON_OFFSET_FILL);

        forwardShader->unbind();
    }
}

template <typename ShaderType>
void BasicAsset<ShaderType>::renderDepth(Camera* cam, const mat4& model)
{
    (void)cam;
    if(depthshader->bind())
    {
        depthshader->uploadModel(model);
        if (unified_buffer)
        {
            unified_buffer->Bind();
            unified_buffer->Draw();
            unified_buffer->Unbind();
        }
        depthshader->unbind();
    }
}

template <typename ShaderType>
void BasicAsset<ShaderType>::renderWireframe(Camera* cam, const mat4& model)
{
    (void)cam;
    if(wireframeshader->bind())
    {
        wireframeshader->uploadModel(model);

        //    glEnable(GL_POLYGON_OFFSET_LINE);

        // negative values shifts the wireframe towards the camera,
        // but a non zero factors does strange things for lines and increases
        // the depth on lines with high slope towards the camera by too much.
        // a visually better solution is to shift the triangles back a bit glPolygonOffset(1,1);
        // and draw the wireframe without polygon offset.
        //    glPolygonOffset(0.0f,-500.0f);

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        if (unified_buffer)
        {
            unified_buffer->Bind();
            unified_buffer->Draw();
            unified_buffer->Unbind();
        }
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        //    glDisable(GL_POLYGON_OFFSET_LINE);

        wireframeshader->unbind();
    }
}

template < typename ShaderType>
void BasicAsset<ShaderType>::renderRaw()
{
    if (unified_buffer)
    {
        unified_buffer->Bind();
        unified_buffer->Draw();
        unified_buffer->Unbind();
    }
}

template < typename ShaderType>
void BasicAsset<ShaderType>::setShader(std::shared_ptr<ShaderType> _shader,
                                                  std::shared_ptr<ShaderType> _forwardShader,
                                                  std::shared_ptr<ShaderType> _depthshader,
                                                  std::shared_ptr<ShaderType> _wireframeshader)
{
    this->deferredShader  = _shader;
    this->forwardShader   = _forwardShader;
    this->depthshader     = _depthshader;
    this->wireframeshader = _wireframeshader;
}

}  // namespace Saiga
