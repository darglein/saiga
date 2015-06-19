#include "framework.h"
#include "opengl/shaderLoader.h"
#include "rendering/deferred_renderer.h"
#include "window/window.h"
void initFramework(Window *window)
{
    ShaderLoader::instance()->addPath("shader");
    ShaderLoader::instance()->addPath("shader/geometry");
    ShaderLoader::instance()->addPath("shader/lighting");
    ShaderLoader::instance()->addPath("shader/post_processing");


    DeferredShader* def = ShaderLoader::instance()->load<DeferredShader>("deferred_mixer.glsl");



    Deferred_Renderer* renderer = new Deferred_Renderer();
    renderer->init(def,window->getWidth(),window->getHeight());
    //    renderer->lighting.setShader(shaderLoader.load<SpotLightShader>("deferred_lighting_spotlight.glsl"));
    renderer->lighting.setShader(ShaderLoader::instance()->load<SpotLightShader>("deferred_lighting_spotlight_shadow.glsl"));
    renderer->lighting.setShader(ShaderLoader::instance()->load<PointLightShader>("deferred_lighting_pointlight_shadow.glsl"));
    //    renderer->lighting.setShader(shaderLoader.load<DirectionalLightShader>("deferred_lighting_directional.glsl"));
    renderer->lighting.setShader(ShaderLoader::instance()->load<DirectionalLightShader>("deferred_lighting_directional_shadow.glsl"));

    renderer->lighting.setDebugShader(ShaderLoader::instance()->load<MVPColorShader>("deferred_lighting_debug.glsl"));
    renderer->lighting.setStencilShader(ShaderLoader::instance()->load<MVPShader>("deferred_lighting_stencil.glsl"));



    //    renderer->postProcessingShader  = shaderLoader.load<PostProcessingShader>("fxaa.glsl");
    //    renderer->postProcessingShader  = shaderLoader.load<PostProcessingShader>("SMAA.glsl");
    //    renderer->postProcessingShader  = shaderLoader.load<PostProcessingShader>("gaussian_blur.glsl");


    renderer->ssaoShader  =  ShaderLoader::instance()->load<SSAOShader>("ssao.glsl");
    renderer->ssao = true;
//    renderer->otherShader  =  window->loadShader<PostProcessingShader>("post_processing.glsl");

    renderer->postProcessingShader = ShaderLoader::instance()->load<PostProcessingShader>("post_processing.glsl");

    renderer->lighting.setRenderDebug( false);

    window->renderer = renderer;
}

