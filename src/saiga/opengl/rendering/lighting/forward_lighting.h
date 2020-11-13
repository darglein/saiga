/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/camera/camera.h"
#include "saiga/core/math/math.h"
#include "saiga/core/window/Interfaces.h"
#include "saiga/opengl/rendering/lighting/renderer_lighting.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/uniformBuffer.h"

namespace Saiga
{
#define MAX_PL_COUNT 1024
#define MAX_SL_COUNT 750
#define MAX_BL_COUNT 1024
#define MAX_DL_COUNT 1024

struct PointLightData
{
    vec4 position;       // xyz, w unused
    vec4 colorDiffuse;   // rgb intensity
    vec4 colorSpecular;  // rgb specular intensity
    vec4 attenuation;    // xyz radius
};

struct SpotLightData
{
    vec4 position;       // xyz, w angle
    vec4 colorDiffuse;   // rgb intensity
    vec4 colorSpecular;  // rgb specular intensity
    vec4 attenuation;    // xyz radius
    vec4 direction;      // xyzw
};

struct BoxLightData
{
    vec4 position;       // xyz, w unused
    vec4 colorDiffuse;   // rgb intensity
    vec4 colorSpecular;  // rgb specular intensity
};

struct DirectionalLightData
{
    vec4 position;       // xyz, w unused
    vec4 colorDiffuse;   // rgb intensity
    vec4 colorSpecular;  // rgb specular intensity
    vec4 direction;      // xyz, w unused
};

struct LightData
{
    PointLightData pointLights[MAX_PL_COUNT];
    SpotLightData spotLights[MAX_SL_COUNT];
    BoxLightData boxLights[MAX_BL_COUNT];
    DirectionalLightData directionalLights[MAX_DL_COUNT];
};

struct LightInfo
{
    int pointLightCount;
    int spotLightCount;
    int boxLightCount;
    int directionalLightCount;
};

#define POINT_LIGHT_DATA_BINDING_POINT 2
#define SPOT_LIGHT_DATA_BINDING_POINT 3
#define BOX_LIGHT_DATA_BINDING_POINT 4
#define DIRECTIONAL_LIGHT_DATA_BINDING_POINT 5
#define LIGHT_INFO_BINDING_POINT 6

class SAIGA_OPENGL_API MVPColorShaderFL : public MVPColorShader
{
   public:
    GLint location_lightDataBlockPoint;
    GLint location_lightDataBlockSpot;
    GLint location_lightDataBlockBox;
    GLint location_lightDataBlockDirectional;
    GLint location_lightInfoBlock;

    virtual void checkUniforms() override
    {
        MVPColorShader::checkUniforms();
        location_lightDataBlockPoint = getUniformBlockLocation("lightDataBlockPoint");
        if (location_lightDataBlockPoint != GL_INVALID_INDEX) setUniformBlockBinding(location_lightDataBlockPoint, POINT_LIGHT_DATA_BINDING_POINT);
        location_lightDataBlockSpot = getUniformBlockLocation("lightDataBlockSpot");
        if (location_lightDataBlockSpot != GL_INVALID_INDEX) setUniformBlockBinding(location_lightDataBlockSpot, SPOT_LIGHT_DATA_BINDING_POINT);
        location_lightDataBlockBox = getUniformBlockLocation("lightDataBlockBox");
        if (location_lightDataBlockBox != GL_INVALID_INDEX) setUniformBlockBinding(location_lightDataBlockBox, BOX_LIGHT_DATA_BINDING_POINT);
        location_lightDataBlockDirectional = getUniformBlockLocation("lightDataBlockDirectional");
        if (location_lightDataBlockDirectional != GL_INVALID_INDEX) setUniformBlockBinding(location_lightDataBlockDirectional, DIRECTIONAL_LIGHT_DATA_BINDING_POINT);


        location_lightInfoBlock = getUniformBlockLocation("lightInfoBlock");
        setUniformBlockBinding(location_lightInfoBlock, LIGHT_INFO_BINDING_POINT);
    }
};

class SAIGA_OPENGL_API ForwardLighting : public RendererLighting
{
   public:
    UniformBuffer lightDataBufferPoint;
    UniformBuffer lightDataBufferSpot;
    UniformBuffer lightDataBufferBox;
    UniformBuffer lightDataBufferDirectional;
    UniformBuffer lightInfoBuffer;

    ForwardLighting();
    ~ForwardLighting();

    void initRender() override;

    void render(Camera* cam, const ViewPort& viewPort) override;

    void renderImGui(bool* p_open = NULL) override;
};

}  // namespace Saiga
