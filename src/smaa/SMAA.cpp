#include "saiga/smaa/SMAA.h"
#include "saiga/smaa/AreaTex.h"
#include "saiga/smaa/SearchTex.h"

#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/geometry/triangle_mesh_generator.h"
#include "saiga/opengl/texture/imageGenerator.h"
#include "saiga/rendering/gbuffer.h"

SMAA::SMAA(int w, int h)
{
    screenSize = vec2(w,h);

    edgesTex = new Texture();
    edgesTex->createEmptyTexture(w,h,GL_RGBA,GL_RGBA8,GL_UNSIGNED_BYTE);

    blendTex = new Texture();
    blendTex->createEmptyTexture(w,h,GL_RGBA,GL_RGBA8,GL_UNSIGNED_BYTE);

    areaTex = new Texture();
    areaTex->createTexture(AREATEX_WIDTH,AREATEX_HEIGHT,GL_RG,GL_RG8,GL_UNSIGNED_BYTE,areaTexBytes);

    searchTex = new Texture();
    searchTex->createTexture(SEARCHTEX_WIDTH,SEARCHTEX_HEIGHT,GL_RED,GL_R8,GL_UNSIGNED_BYTE,searchTexBytes);


    auto qb = TriangleMeshGenerator::createFullScreenQuadMesh();
    qb->createBuffers(quadMesh);

    assert_no_glerror();

//    ssaoShader  =  ShaderLoader::instance()->load<SSAOShader>("post_processing/ssao2.glsl");
//    blurShader = ShaderLoader::instance()->load<MVPTextureShader>("post_processing/ssao_blur.glsl");

//    setKernelSize(32);

//    auto randomImage = ImageGenerator::randomNormalized(32,32);
//    randomTexture = std::make_shared<Texture>();
//    randomTexture->fromImage(*randomImage);
//    randomTexture->setWrap(GL_REPEAT);

//    clearSSAO();
}
