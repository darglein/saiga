#include "saiga/world/proceduralSkybox.h"
#include "saiga/geometry/triangle_mesh_generator.h"
#include "saiga/opengl/shader/shaderLoader.h"


ProceduralSkybox::ProceduralSkybox(){

    auto sb = TriangleMeshGenerator::createFullScreenQuadMesh();
    sb->transform(glm::translate(mat4(),vec3(0,0,1-glm::epsilon<float>())));
    sb->createBuffers(mesh);

    shader = ShaderLoader::instance()->load<MVPShader>("geometry/proceduralSkybox.glsl");
}


void ProceduralSkybox::render(Camera* cam){
    shader->bind();
    shader->uploadAll(model,cam->view,cam->proj);
    mesh.bindAndDraw();

    shader->unbind();
}
