/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/world/proceduralSkybox.h"
#include "saiga/geometry/triangle_mesh_generator.h"
#include "saiga/opengl/shader/shaderLoader.h"

namespace Saiga {

void ProceduralSkyboxShader::checkUniforms()
{
    MVPShader::checkUniforms();
    location_params = Shader::getUniformLocation("params");
}

void ProceduralSkyboxShader::uploadParams(float horizonHeight, float distance)
{
    vec4 params = vec4(horizonHeight,distance,0,0);
    Shader::upload(location_params,params);

}



ProceduralSkybox::ProceduralSkybox(){

    auto sb = TriangleMeshGenerator::createFullScreenQuadMesh();

    sb->transform(glm::translate(mat4(1),vec3(0,0,1-glm::epsilon<float>())));

//    sb->createBuffers(mesh);
    mesh.fromMesh(*sb);

    shader = ShaderLoader::instance()->load<ProceduralSkyboxShader>("geometry/proceduralSkybox.glsl");
}


void ProceduralSkybox::render(Camera* cam){
    shader->bind();
    shader->uploadModel(model);
    shader->uploadParams(horizonHeight,distance);
    mesh.bindAndDraw();

    shader->unbind();
}

}
