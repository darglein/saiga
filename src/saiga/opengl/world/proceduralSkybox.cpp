/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/world/proceduralSkybox.h"

#include "saiga/core/geometry/triangle_mesh_generator.h"
#include "saiga/opengl/shader/shaderLoader.h"

namespace Saiga
{
void ProceduralSkyboxShader::checkUniforms()
{
    MVPShader::checkUniforms();
    location_params = Shader::getUniformLocation("params");
}

void ProceduralSkyboxShader::uploadParams(vec3 sunDir, float horizonHeight, float distance, float sunIntensity,
                                          float sunSize)
{
    vec4 params = vec4(horizonHeight, distance, sunIntensity, sunSize);
    Shader::upload(0, params);
    Shader::upload(1, sunDir);
}



ProceduralSkybox::ProceduralSkybox()
{
    auto sb = TriangleMeshGenerator::createFullScreenQuadMesh();

    sb->transform(translate(vec3(0, 0, 1 - epsilon<float>())));

    //    sb->createBuffers(mesh);
    mesh.fromMesh(*sb);

    shader = shaderLoader.load<ProceduralSkyboxShader>("geometry/proceduralSkybox.glsl");
}


void ProceduralSkybox::render(Camera* cam)
{
    shader->bind();
    shader->uploadModel(model);
    shader->uploadParams(sunDir, horizonHeight, distance, sunIntensity, sunSize);
    mesh.bindAndDraw();

    shader->unbind();
}



}  // namespace Saiga
