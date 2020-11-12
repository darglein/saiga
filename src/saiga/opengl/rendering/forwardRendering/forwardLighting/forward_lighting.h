/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/camera/camera.h"
#include "saiga/core/math/math.h"
#include "saiga/core/window/Interfaces.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/uniformBuffer.h"

#include "saiga/opengl/rendering/lighting/point_light.h"

namespace Saiga
{

#define MAX_PL_COUNT 1024

struct LightData
{
    vec4 plPositions[MAX_PL_COUNT];
    vec4 plColors[MAX_PL_COUNT];
    vec4 plAttenuations[MAX_PL_COUNT];
    int plCount;
};

#define LIGHT_DATA_BINDING_POINT 2

class SAIGA_OPENGL_API MVPColorShaderFL : public MVPColorShader
{
   public:
    GLint location_lightDataBlock;

    virtual void checkUniforms() override
    {
        MVPColorShader::checkUniforms();
        location_lightDataBlock = getUniformBlockLocation("lightDataBlock");
        setUniformBlockBinding(location_lightDataBlock, LIGHT_DATA_BINDING_POINT);
    }
};

class SAIGA_OPENGL_API ForwardLighting
{
   public:
    vec4 clearColor;
    int lightCount;

    UniformBuffer pointLightBuffer;

    bool debugDraw;

    ForwardLighting();
    ~ForwardLighting();

    void init(int width, int height);
    void resize(int width, int height);

    void addPointLight(std::shared_ptr<PointLight> light);
    void removePointLight(std::shared_ptr<PointLight> light);
    std::vector<std::shared_ptr<PointLight> > pointLights;

    void render(RenderingInterface* renderingInterface, Camera* camera);


    void printTimings();
    void renderImGui(bool* p_open = NULL);


    int width;
    int height;
};

}  // namespace Saiga
