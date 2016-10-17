#include "saiga/smaa/SMAA.h"
#include "saiga/smaa/AreaTex.h"
#include "saiga/smaa/SearchTex.h"

#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/geometry/triangle_mesh_generator.h"
#include "saiga/opengl/texture/imageGenerator.h"
#include "saiga/rendering/gbuffer.h"

void SMAABlendingWeightCalculationShader::checkUniforms()
{
    Shader::checkUniforms();
    location_edgeTex = Shader::getUniformLocation("edgeTex");
    location_areaTex = Shader::getUniformLocation("areaTex");
    location_searchTex = Shader::getUniformLocation("searchTex");

}

void SMAABlendingWeightCalculationShader::uploadTextures(raw_Texture *edgeTex, raw_Texture *areaTex, raw_Texture *searchTex)
{
    edgeTex->bind(0);
    Shader::upload(location_edgeTex,0);

    areaTex->bind(1);
    Shader::upload(location_areaTex,1);

    searchTex->bind(2);
    Shader::upload(location_searchTex,2);
}

void SMAANeighborhoodBlendingShader::checkUniforms()
{
    Shader::checkUniforms();
    location_colorTex = Shader::getUniformLocation("colorTex");
    location_blendTex = Shader::getUniformLocation("blendTex");
}

void SMAANeighborhoodBlendingShader::uploadTextures(raw_Texture *colorTex, raw_Texture *blendTex)
{
    colorTex->bind(0);
    Shader::upload(location_colorTex,0);

    blendTex->bind(1);
    Shader::upload(location_blendTex,1);
}


SMAA::SMAA(int w, int h)
{
    screenSize = vec2(w,h);

    edgesTex = new Texture();
    edgesTex->createEmptyTexture(w,h,GL_RGBA,GL_RGBA8,GL_UNSIGNED_BYTE);
    edgesFb.create();
    edgesFb.attachTexture( framebuffer_texture_t(edgesTex) );
    edgesFb.drawToAll();
    edgesFb.check();
    edgesFb.unbind();

    blendTex = new Texture();
    blendTex->createEmptyTexture(w,h,GL_RGBA,GL_RGBA8,GL_UNSIGNED_BYTE);
    blendFb.create();
    blendFb.attachTexture( framebuffer_texture_t(edgesTex) );
    blendFb.drawToAll();
    blendFb.check();
    blendFb.unbind();

    areaTex = new Texture();
    areaTex->createTexture(AREATEX_WIDTH,AREATEX_HEIGHT,GL_RG,GL_RG8,GL_UNSIGNED_BYTE,areaTexBytes);

    searchTex = new Texture();
    searchTex->createTexture(SEARCHTEX_WIDTH,SEARCHTEX_HEIGHT,GL_RED,GL_R8,GL_UNSIGNED_BYTE,searchTexBytes);


    smaaEdgeDetectionShader = ShaderLoader::instance()->load<PostProcessingShader>("post_processing/smaa/SMAAEdgeDetection.glsl");
    smaaBlendingWeightCalculationShader = ShaderLoader::instance()->load<SMAABlendingWeightCalculationShader>("post_processing/smaa/SMAABlendingWeightCalculation.glsl");
    smaaNeighborhoodBlendingShader = ShaderLoader::instance()->load<SMAANeighborhoodBlendingShader>("post_processing/smaa/SMAANeighborhoodBlending.glsl");


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

void SMAA::render(framebuffer_texture_t input, Framebuffer &output)
{


    glDisable( GL_FRAMEBUFFER_SRGB );

    glViewport(0, 0, screenSize.x,screenSize.y);

    edgesFb.bind();

    glClear(GL_COLOR_BUFFER_BIT);
    smaaEdgeDetectionShader->bind();
    vec4 ss(screenSize.x,screenSize.y,1.0/screenSize.x,1.0/screenSize.y);
    smaaEdgeDetectionShader->uploadScreenSize(ss);
    smaaEdgeDetectionShader->uploadTexture(input.get() );
    quadMesh.bindAndDraw();
    smaaEdgeDetectionShader->unbind();
    assert_no_glerror();

    blendFb.bind();
    glClear(GL_COLOR_BUFFER_BIT);
    smaaBlendingWeightCalculationShader->bind();
    smaaBlendingWeightCalculationShader->uploadTextures(edgesTex,areaTex,searchTex);
    quadMesh.bindAndDraw();
    smaaBlendingWeightCalculationShader->unbind();
    assert_no_glerror();


    output.bind();
    glClear(GL_COLOR_BUFFER_BIT);
    smaaNeighborhoodBlendingShader->bind();
    smaaNeighborhoodBlendingShader->uploadTextures(input.get(),blendTex);
    quadMesh.bindAndDraw();
    smaaNeighborhoodBlendingShader->unbind();
    assert_no_glerror();

}




