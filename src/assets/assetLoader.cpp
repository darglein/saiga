#include "saiga/assets/assetLoader.h"

#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/texture/textureLoader.h"




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
    texturedAssetDepthShader = ShaderLoader::instance()->load<MVPTextureShader>("texturedAsset.glsl");
    texturedAssetWireframeShader = ShaderLoader::instance()->load<MVPTextureShader>("texturedAsset.glsl");
}
