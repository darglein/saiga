#include "saiga/world/skybox.h"
#include "saiga/geometry/triangle_mesh_generator.h"




Skybox::Skybox(){
    aabb bb(vec3(-1),vec3(1));
    auto sb = TriangleMeshGenerator::createSkyboxMesh(bb);
    sb->createBuffers(mesh);
}

void Skybox::setPosition(const vec3& p){
    model[3] = vec4(p.x,0,p.z,1);
}

void Skybox::setDistance(float d){
    model[0][0] = d;
    model[1][1] = d;
    model[2][2] = d;
}


void Skybox::render(Camera* cam){
    shader->bind();
    shader->uploadAll(cam,model);
    shader->uploadTexture(cube_texture);
    mesh.bindAndDraw();
    cube_texture->unbind();

    shader->unbind();
}
