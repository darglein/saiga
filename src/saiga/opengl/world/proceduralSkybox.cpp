/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/world/proceduralSkybox.h"

#include "saiga/core/model/model_from_shape.h"
#include "saiga/opengl/shader/shaderLoader.h"

namespace Saiga
{
ProceduralSkybox::ProceduralSkybox(const std::string& shader_str)
{
    mesh.fromMesh(FullScreenQuad().transform(translate(vec3(0, 0, 1 - epsilon<float>()))));
    shader = shaderLoader.load<MVPShader>(shader_str);
}


void ProceduralSkybox::render(Camera* cam, const mat4& model)
{
    shader->bind();
    shader->uploadModel(model);

    vec4 params = vec4(horizonHeight, distance, sunIntensity, sunSize);
    shader->upload(0, params);
    shader->upload(1, sunDir);
    shader->upload(2, sunColor);
    shader->upload(3, highSkyColor);
    shader->upload(4, lowSkyColor);
    mesh.bindAndDraw();

    shader->unbind();
}



}  // namespace Saiga
