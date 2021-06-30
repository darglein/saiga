/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/world/proceduralSkybox.h"

#include "saiga/core/model/model_from_shape.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/core/util/color.h"
namespace Saiga
{
ProceduralSkybox::ProceduralSkybox(const std::string& shader_str)
 : mesh(FullScreenQuad().transform(translate(vec3(0, 0, 1 - epsilon<float>()))))
{
    shader = shaderLoader.load<MVPShader>(shader_str);
}


void ProceduralSkybox::render(Camera* cam, const mat4& model)
{
    if(shader->bind())
    {
        shader->uploadModel(model);

        vec4 params = vec4(horizonHeight, distance, sunIntensity, sunSize);
        shader->upload(0, params);

        shader->upload(1, Color::srgb2linearrgb(sunDir));
        shader->upload(2, Color::srgb2linearrgb(sunColor));
        shader->upload(3, Color::srgb2linearrgb(highSkyColor));
        shader->upload(4, Color::srgb2linearrgb(lowSkyColor));
        mesh.BindAndDraw();

        shader->unbind();
    }
}



}  // namespace Saiga
