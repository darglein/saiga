#include "saiga/assets/assetLoader.h"

#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/texture/textureLoader.h"

#include "saiga/geometry/triangle_mesh_generator.h"
#include "saiga/opengl/texture/imageGenerator.h"


AssetLoader2::AssetLoader2()
{
     loadDefaultShaders();
}

AssetLoader2::~AssetLoader2()
{

}


void AssetLoader2::loadDefaultShaders()
{
    basicAssetShader = ShaderLoader::instance()->load<MVPShader>("deferred_mvp_model.glsl");
    basicAssetDepthshader = ShaderLoader::instance()->load<MVPShader>("deferred_mvp_model_depth.glsl");
    basicAssetWireframeShader = ShaderLoader::instance()->load<MVPShader>("deferred_mvp_model_wireframe.glsl");

    texturedAssetShader = ShaderLoader::instance()->load<MVPTextureShader>("texturedAsset.glsl");
    texturedAssetDepthShader = ShaderLoader::instance()->load<MVPTextureShader>("texturedAsset_depth.glsl");
    texturedAssetWireframeShader = ShaderLoader::instance()->load<MVPTextureShader>("texturedAsset_wireframe.glsl");

    animatedAssetShader = ShaderLoader::instance()->load<BoneShader>("deferred_mvp_bones.glsl");
    animatedAssetDepthshader = ShaderLoader::instance()->load<BoneShader>("deferred_mvp_bones_depth.glsl");
    animatedAssetWireframeShader = ShaderLoader::instance()->load<BoneShader>("deferred_mvp_bones.glsl");
}

TexturedAsset *AssetLoader2::loadDebugPlaneAsset(vec2 size, float quadSize, Color color1, Color color2)
{
    auto plainMesh = TriangleMeshGenerator::createMesh(Plane());


    mat4 scale = glm::scale(mat4(1),vec3(size.x,1,size.y));
    plainMesh->transform(scale);

    for(auto& v : plainMesh->vertices){
        v.texture *= size / quadSize;
    }

    TexturedAsset* plainAsset = new TexturedAsset();

    plainAsset->mesh.addMesh(*plainMesh);

    for(auto v : plainAsset->mesh.vertices)
        cout << v.position << endl;

    TexturedAsset::TextureGroup tg;
    tg.startIndex = 0;
    tg.indices = plainMesh->numIndices();


    auto cbImage = ImageGenerator::checkerBoard(color1,color2,16,2,2);
    Texture* cbTexture = new Texture();
    cbTexture->fromImage(*cbImage);
    tg.texture = cbTexture;
//    tg.texture = TextureLoader::instance()->load("debug2x2.png");
    tg.texture->setFiltering(GL_NEAREST);
    tg.texture->setWrap(GL_REPEAT);
    tg.texture->generateMipmaps();
    //    tg.texture->setFiltering(GL_NEAREST_MIPMAP_LINEAR);
    plainAsset->groups.push_back(tg);


    plainAsset->create("test",texturedAssetShader,texturedAssetDepthShader,texturedAssetWireframeShader);

    return plainAsset;
}
