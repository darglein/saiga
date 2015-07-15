#include "saiga/framework.h"
#include "saiga/rendering/deferred_renderer.h"
#include "saiga/window/window.h"

#include "saiga/opengl/shaderLoader.h"
#include "saiga/opengl/objloader.h"
#include "saiga/opengl/texture/textureLoader.h"


#include "saiga/rendering/lighting/directional_light.h"
#include "saiga/rendering/lighting/point_light.h"
#include "saiga/rendering/lighting/spot_light.h"
#include "saiga/rendering/lighting/box_light.h"

#include "saiga/util/configloader.h"


std::string SHADER_PATH;
std::string TEXTURE_PATH;
std::string MATERIAL_PATH;
std::string OBJ_PATH;

void readConfigFile(){
    ConfigLoader cl;
    cl.loadFile("saiga-config.txt");

    SHADER_PATH = cl.getString("SHADER_PATH","/usr/local/share/saiga/shader");
    TEXTURE_PATH = cl.getString("TEXTURE_PATH","textures");
    MATERIAL_PATH = cl.getString("MATERIAL_PATH","objs");
    OBJ_PATH = cl.getString("OBJ_PATH","objs");

}


void initFramework(Window *window)
{
    readConfigFile();

    ShaderLoader::instance()->addPath(SHADER_PATH);
    ShaderLoader::instance()->addPath(SHADER_PATH+"/geometry");
    ShaderLoader::instance()->addPath(SHADER_PATH+"/lighting");
    ShaderLoader::instance()->addPath(SHADER_PATH+"/post_processing");

    TextureLoader::instance()->addPath(TEXTURE_PATH);
    TextureLoader::instance()->addPath(OBJ_PATH);
    TextureLoader::instance()->addPath(".");

    MaterialLoader::instance()->addPath(".");
    MaterialLoader::instance()->addPath(OBJ_PATH);

    ObjLoader::instance()->addPath(".");
    ObjLoader::instance()->addPath(OBJ_PATH);


    DeferredShader* def = ShaderLoader::instance()->load<DeferredShader>("deferred_mixer.glsl");



    Deferred_Renderer* renderer = new Deferred_Renderer();
    renderer->init(def,window->getWidth(),window->getHeight());
    //    renderer->lighting.setShader(shaderLoader.load<SpotLightShader>("deferred_lighting_spotlight.glsl"));

    Shader::ShaderCodeInjections shadowInjection;
    shadowInjection.emplace_back(GL_FRAGMENT_SHADER,
                                    "#define SHADOWS",1); //after the version number

    renderer->lighting.setShader(
                ShaderLoader::instance()->load<SpotLightShader>("deferred_lighting_spotlight_shadow.glsl"),
                ShaderLoader::instance()->load<SpotLightShader>("deferred_lighting_spotlight_shadow.glsl",shadowInjection)
                );

    renderer->lighting.setShader(
                ShaderLoader::instance()->load<PointLightShader>("deferred_lighting_pointlight_shadow.glsl"),
                ShaderLoader::instance()->load<PointLightShader>("deferred_lighting_pointlight_shadow.glsl",shadowInjection)
                );

    renderer->lighting.setShader(
                ShaderLoader::instance()->load<DirectionalLightShader>("deferred_lighting_directional_shadow.glsl"),
                ShaderLoader::instance()->load<DirectionalLightShader>("deferred_lighting_directional_shadow.glsl",shadowInjection)
                                 );

    renderer->lighting.setShader(
                ShaderLoader::instance()->load<BoxLightShader>("deferred_lighting_boxlight_shadow.glsl"),
                ShaderLoader::instance()->load<BoxLightShader>("deferred_lighting_boxlight_shadow.glsl",shadowInjection)
                                 );

    renderer->lighting.setDebugShader(ShaderLoader::instance()->load<MVPColorShader>("deferred_lighting_debug.glsl"));
    renderer->lighting.setStencilShader(ShaderLoader::instance()->load<MVPShader>("deferred_lighting_stencil.glsl"));

    renderer->lighting.loadShaders();


    //    renderer->postProcessingShader  = shaderLoader.load<PostProcessingShader>("fxaa.glsl");
    //    renderer->postProcessingShader  = shaderLoader.load<PostProcessingShader>("SMAA.glsl");
    //    renderer->postProcessingShader  = shaderLoader.load<PostProcessingShader>("gaussian_blur.glsl");


    renderer->ssaoShader  =  ShaderLoader::instance()->load<SSAOShader>("ssao.glsl");
    renderer->ssao = true;
//    renderer->otherShader  =  ShaderLoader::instance()->load<PostProcessingShader>("post_processing.glsl");

    PostProcessingShader* pps = ShaderLoader::instance()->load<PostProcessingShader>("post_processing.glsl");
    std::vector<PostProcessingShader*> defaultEffects;
    defaultEffects.push_back(pps);

    renderer->postProcessor.setPostProcessingEffects(defaultEffects);
//    renderer->postProcessor.postProcessingEffects.push_back(renderer->postProcessingShader);

//    PostProcessingShader* bla = ShaderLoader::instance()->load<PostProcessingShader>("gaussian_blur.glsl");
//    renderer->postProcessor.postProcessingEffects.push_back(bla);
//    renderer->postProcessor.postProcessingEffects.push_back(bla);
//    renderer->postProcessor.postProcessingEffects.push_back(bla);
//    renderer->postProcessor.postProcessingEffects.push_back(bla);

    renderer->lighting.setRenderDebug( false);

    window->renderer = renderer;

    cout<<"========================== Framework initialization done! =========================="<<endl;
}

