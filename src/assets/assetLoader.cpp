/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/assets/assetLoader.h"

#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/texture/textureLoader.h"

#include "saiga/geometry/triangle_mesh_generator.h"
#include "saiga/image/imageGenerator.h"

namespace Saiga {

AssetLoader2::AssetLoader2()
{
     loadDefaultShaders();
}

AssetLoader2::~AssetLoader2()
{

}


void AssetLoader2::loadDefaultShaders()
{
    basicAssetShader = ShaderLoader::instance()->load<MVPShader>("geometry/deferred_mvp_model.glsl");
    basicAssetDepthshader = ShaderLoader::instance()->load<MVPShader>("geometry/deferred_mvp_model_depth.glsl");
    basicAssetWireframeShader = ShaderLoader::instance()->load<MVPShader>("geometry/deferred_mvp_model_wireframe.glsl");

    texturedAssetShader = ShaderLoader::instance()->load<MVPTextureShader>("geometry/texturedAsset.glsl");
    texturedAssetDepthShader = ShaderLoader::instance()->load<MVPTextureShader>("geometry/texturedAsset_depth.glsl");
    texturedAssetWireframeShader = ShaderLoader::instance()->load<MVPTextureShader>("geometry/texturedAsset_wireframe.glsl");

    animatedAssetShader = ShaderLoader::instance()->load<BoneShader>("geometry/deferred_mvp_bones.glsl");
    animatedAssetDepthshader = ShaderLoader::instance()->load<BoneShader>("geometry/deferred_mvp_bones_depth.glsl");
    animatedAssetWireframeShader = ShaderLoader::instance()->load<BoneShader>("geometry/deferred_mvp_bones.glsl");
}

std::shared_ptr<TexturedAsset> AssetLoader2::loadDebugPlaneAsset(vec2 size, float quadSize, Color color1, Color color2)
{
        auto cbImage = ImageGenerator::checkerBoard(color1,color2,16,2,2);
        std::shared_ptr<Texture> cbTexture = std::make_shared<Texture>();
        cbTexture->fromImage(*cbImage);
        cbTexture->setFiltering(GL_NEAREST);
        cbTexture->setWrap(GL_REPEAT);
        cbTexture->generateMipmaps();
        std::shared_ptr<TexturedAsset> asset = loadDebugTexturedPlane(cbTexture,size);
        for(auto &v : asset->mesh.vertices){
            v.texture *= size / quadSize;
        }
        asset->mesh.createBuffers(asset->buffer);
        return asset;
}

std::shared_ptr<TexturedAsset> AssetLoader2::loadDebugTexturedPlane(std::shared_ptr<Texture> texture, vec2 size)
{
    auto plainMesh = TriangleMeshGenerator::createMesh(Plane());
    mat4 scale = glm::scale(mat4(1),vec3(size.x,1,size.y));
    plainMesh->transform(scale);

    auto asset = std::make_shared<TexturedAsset>();

    asset->mesh.addMesh(*plainMesh);

    for(auto& v : asset->mesh.vertices){
        v.data = vec4(0.5,0,0,0);
    }

    TexturedAsset::TextureGroup tg;
    tg.startIndex = 0;
    tg.indices = plainMesh->numIndices();
    tg.texture = texture;
    asset->groups.push_back(tg);
    asset->create("Plane",texturedAssetShader,texturedAssetDepthShader,texturedAssetWireframeShader);

    return asset;
}

std::shared_ptr<ColoredAsset> AssetLoader2::loadDebugArrow(float radius, float length, vec4 color)
{
//    auto plainMesh = TriangleMeshGenerator::createMesh(Plane());
    auto cylinderMesh = TriangleMeshGenerator::createCylinderMesh(radius,length,12);
    mat4 m = glm::translate(mat4(),vec3(0,length*0.5f,0));
    cylinderMesh->transform(m);

    float coneH = length * 0.3f;
    float coneR = radius * 1.3f;
    auto coneMesh = TriangleMeshGenerator::createMesh(Cone(vec3(0),vec3(0,1,0),coneR,coneH),12);
    m = glm::translate(mat4(),vec3(0,length+coneH,0));
        coneMesh->transform(m);

    auto asset = std::make_shared<ColoredAsset>();
    asset->mesh.addMesh(*cylinderMesh);
    asset->mesh.addMesh(*coneMesh);

    for(auto& v : asset->mesh.vertices){
        v.color = color;
        v.data = vec4(0.5,0,0,0);
    }

    asset->create("Arrow",basicAssetShader,basicAssetDepthshader,basicAssetWireframeShader);
    return asset;
}

std::shared_ptr<ColoredAsset> AssetLoader2::assetFromMesh(std::shared_ptr<TriangleMesh<VertexNT, GLuint> > mesh, const vec4 &color)
{

    auto asset = std::make_shared<ColoredAsset>();
    asset->mesh.addMesh(*mesh);

    for(auto& v : asset->mesh.vertices){
        v.color = color;
        v.data = vec4(0.5,0,0,0);
    }

    asset->create("Fromsdfg",basicAssetShader,basicAssetDepthshader,basicAssetWireframeShader);
    return asset;
}

}
