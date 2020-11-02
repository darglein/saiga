/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/math.h"
#include "saiga/opengl/shader/basic_shaders.h"

namespace Saiga
{

struct StandardPointLightParameters // for now
{
};
struct StandardPointLight // for now
{
    StandardPointLight(const StandardPointLightParameters& params) {}
    vec3 position = make_vec3(0.0);
    float intensity = 600.0f;
};

class SAIGA_OPENGL_API BasicForwardLighting
{
   public:
    vec4 clearColor;
    int lightCount;
    int visibleLights;

    bool debugDraw;

    BasicForwardLighting();
    ~BasicForwardLighting();

    void init(int width, int height);
    void resize(int width, int height);

    std::shared_ptr<StandardPointLight> addPointLight(StandardPointLightParameters params);
    void removePointLight(std::shared_ptr<StandardPointLight> light);
    std::vector<std::shared_ptr<StandardPointLight> > pointLights;

    void initRender();
    void endRender();

    void printTimings();
    void renderImGui(bool* p_open = NULL);


    int width;
    int height;
};

} // namespace Saiga
