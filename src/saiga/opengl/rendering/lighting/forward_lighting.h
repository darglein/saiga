/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/camera/camera.h"
#include "saiga/core/math/math.h"
#include "saiga/core/window/Interfaces.h"
#include "saiga/opengl/rendering/lighting/clusterer.h"
#include "saiga/opengl/rendering/lighting/renderer_lighting.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/shaderStorageBuffer.h"
#include "saiga/opengl/uniformBuffer.h"

namespace Saiga
{
class SAIGA_OPENGL_API MVPColorShaderFL : public MVPColorShader
{
   public:
    GLint location_lightInfoBlock;

    virtual void checkUniforms() override
    {
        MVPColorShader::checkUniforms();

        location_lightInfoBlock = getUniformBlockLocation("lightInfoBlock");
        setUniformBlockBinding(location_lightInfoBlock, LIGHT_INFO_BINDING_POINT);
    }
};

class SAIGA_OPENGL_API MVPTextureShaderFL : public MVPTextureShader
{
   public:
    GLint location_lightInfoBlock;

    virtual void checkUniforms() override
    {
        MVPTextureShader::checkUniforms();

        location_lightInfoBlock = getUniformBlockLocation("lightInfoBlock");
        setUniformBlockBinding(location_lightInfoBlock, LIGHT_INFO_BINDING_POINT);
    }
};

class SAIGA_OPENGL_API ForwardLighting : public RendererLighting
{
   public:

    UniformBuffer lightInfoBuffer;

    ForwardLighting(GLTimerSystem* timer);
    ForwardLighting& operator=(ForwardLighting& l) = delete;
    ~ForwardLighting();

    void init(int _width, int _height, bool _useTimers) override;

    void resize(int _width, int _height) override;

    void initRender() override;

    void cluster(Camera* cam, const ViewPort& viewPort);

    void render(Camera* cam, const ViewPort& viewPort) override;

    void renderImGui() override;

    void setClusterType(int tp) override;

    std::shared_ptr<Clusterer> getClusterer() override { return lightClusterer; };

   public:
    std::shared_ptr<Clusterer> lightClusterer;
    int clustererType = 0;
};

}  // namespace Saiga
