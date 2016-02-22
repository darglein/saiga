#include "saiga/rendering/lighting/deferred_lighting.h"
#include "saiga/util/inputcontroller.h"

#include "saiga/rendering/deferred_renderer.h"
#include "saiga/util/error.h"

#include "saiga/rendering/lighting/directional_light.h"
#include "saiga/rendering/lighting/point_light.h"
#include "saiga/rendering/lighting/spot_light.h"
#include "saiga/rendering/lighting/box_light.h"

#include "saiga/geometry/triangle_mesh_generator.h"
#include "saiga/opengl/texture/cube_texture.h"

#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/rendering/renderer.h"



DeferredLighting::DeferredLighting(GBuffer &framebuffer):gbuffer(framebuffer){
    
    createInputCommands();
    createLightMeshes();


}

DeferredLighting::~DeferredLighting(){
}

void DeferredLighting::loadShaders()
{
    ShaderPart::ShaderCodeInjections shadowInjection;
    shadowInjection.emplace_back(GL_FRAGMENT_SHADER,
                                 "#define SHADOWS",1); //after the version number

    spotLightShader = ShaderLoader::instance()->load<SpotLightShader>("light_spot.glsl");
    spotLightShadowShader = ShaderLoader::instance()->load<SpotLightShader>("light_spot.glsl",shadowInjection);


    pointLightShader = ShaderLoader::instance()->load<PointLightShader>("light_point.glsl");
    pointLightShadowShader = ShaderLoader::instance()->load<PointLightShader>("light_point.glsl",shadowInjection);

    directionalLightShader = ShaderLoader::instance()->load<DirectionalLightShader>("light_directional.glsl");
    directionalLightShadowShader = ShaderLoader::instance()->load<DirectionalLightShader>("light_directional.glsl",shadowInjection);

    boxLightShader = ShaderLoader::instance()->load<BoxLightShader>("light_box.glsl");
    boxLightShadowShader = ShaderLoader::instance()->load<BoxLightShader>("light_box.glsl",shadowInjection);

    debugShader = ShaderLoader::instance()->load<MVPColorShader>("debugmesh.glsl");
    stencilShader = ShaderLoader::instance()->load<MVPShader>("stenciltest.glsl");

    blitDepthShader = ShaderLoader::instance()->load<MVPTextureShader>("blitDepth.glsl");
    lightAccumulationShader = ShaderLoader::instance()->load<LightAccumulationShader>("lightaccumulation.glsl");
}

void DeferredLighting::init(int width, int height){
    this->width=width;this->height=height;


    lightAccumulationBuffer.create();
    Texture* depth_stencil = new Texture();
    depth_stencil->createEmptyTexture(width,height,GL_DEPTH_STENCIL, GL_DEPTH24_STENCIL8,GL_UNSIGNED_INT_24_8);
    lightAccumulationBuffer.attachTextureDepthStencil(depth_stencil);

    lightAccumulationTexture = new Texture();
    lightAccumulationTexture->createEmptyTexture(width,height,GL_RGBA,GL_RGBA16,GL_UNSIGNED_SHORT);
    lightAccumulationBuffer.attachTexture(lightAccumulationTexture);
    glDrawBuffer( GL_COLOR_ATTACHMENT0);
    lightAccumulationBuffer.check();
    lightAccumulationBuffer.unbind();
}

void DeferredLighting::resize(int width, int height)
{
    this->width=width;this->height=height;
    lightAccumulationBuffer.resize(width,height);
}

void DeferredLighting::cullLights(Camera *cam){

    visibleLights = directionalLights.size();

    //cull lights that are not visible
    for(SpotLight* &light : spotLights){
        if(light->isActive()){
            light->calculateCamera();
            light->cam.recalculatePlanes();
            visibleLights += (light->cullLight(cam))? 0 : 1;
        }
    }

    for(BoxLight* &light : boxLights){
        if(light->isActive()){
            light->calculateCamera();
            light->cam.recalculatePlanes();
            visibleLights += (light->cullLight(cam))? 0 : 1;
        }
    }


    for(PointLight* &light : pointLights){
        if(light->isActive()){
            visibleLights += (light->cullLight(cam))? 0 : 1;
        }
    }
}

void DeferredLighting::renderDepthMaps(Program *renderer){
    totalLights = 0;
    renderedDepthmaps = 0;

    totalLights = directionalLights.size() + spotLights.size() + pointLights.size();


    for(DirectionalLight* &light : directionalLights){
        if(light->shouldCalculateShadowMap()){
            renderedDepthmaps++;
            light->bindShadowMap();
            light->cam.recalculatePlanes();
            renderer->renderDepth(&light->cam);
            light->unbindShadowMap();
        }
    }


    for(BoxLight* &light : boxLights){
        if(light->shouldCalculateShadowMap()){
            renderedDepthmaps++;
            light->bindShadowMap();
            light->cam.recalculatePlanes();
            renderer->renderDepth(&light->cam);
            light->unbindShadowMap();
        }
    }

    for(SpotLight* &light : spotLights){
        if(light->shouldCalculateShadowMap()){
            renderedDepthmaps++;
            light->bindShadowMap();
            renderer->renderDepth(&light->cam);
            light->unbindShadowMap();
        }
    }


    for(PointLight* &light : pointLights){

        if(light->shouldCalculateShadowMap()){
            renderedDepthmaps+=6;
            for(int i=0;i<6;i++){
                light->bindFace(i);
                light->calculateCamera(i);
                light->cam.recalculatePlanes();
                renderer->renderDepth(&light->cam);
                light->unbindShadowMap();
            }

        }
    }

}

void DeferredLighting::render(Camera* cam){
//    gbuffer.blitDepth(lightAccumulationBuffer.getId());

    lightAccumulationBuffer.bind();

    //viewport is maybe different after shadow map rendering
    glViewport(0,0,width,height);



//    glClear( GL_STENCIL_BUFFER_BIT );
//    glClear( GL_COLOR_BUFFER_BIT );

//    glDepthMask(GL_FALSE);
//    glDisable(GL_DEPTH_TEST);

    blitGbufferDepthToAccumulationBuffer();
    assert_no_glerror();





    //deferred lighting uses additive blending of the lights.
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_ONE, GL_ONE);

    //never overwrite current depthbuffer
    glDepthMask(GL_FALSE);

    //point- and spot- lights are using stencil culling
    glEnable(GL_STENCIL_TEST);


    renderStencilVolume(spotLightMesh,spotLights); //mark pixels inside the light volume
    setupLightPass();
    renderLightVolume<SpotLight,SpotLightShader,false>(spotLightMesh,spotLights,cam,spotLightShader); //draw back faces without depthtest
    renderLightVolume<SpotLight,SpotLightShader,true>(spotLightMesh,spotLights,cam,spotLightShadowShader);
    assert_no_glerror();


    renderStencilVolume(pointLightMesh,pointLights);  //mark pixels inside the light volume
    setupLightPass();
    renderLightVolume<PointLight,PointLightShader,false>(pointLightMesh,pointLights,cam,pointLightShader); //draw back faces without depthtest
    renderLightVolume<PointLight,PointLightShader,true>(pointLightMesh,pointLights,cam,pointLightShadowShader);
    assert_no_glerror();


    renderStencilVolume(boxLightMesh,boxLights); //mark pixels inside the light volume
    setupLightPass();
    renderLightVolume<BoxLight,BoxLightShader,false>(boxLightMesh,boxLights,cam,boxLightShader); //draw back faces without depthtest
    renderLightVolume<BoxLight,BoxLightShader,true>(boxLightMesh,boxLights,cam,boxLightShadowShader);
    assert_no_glerror();


    glDisable(GL_STENCIL_TEST);
    //use default culling
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    renderDirectionalLights(cam,false);
    renderDirectionalLights(cam,true);

    //reset state
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);

    if(drawDebug){
        glDepthMask(GL_TRUE);
        renderDebug();
        glDepthMask(GL_FALSE);
    }

    lightAccumulationBuffer.unbind();

    assert_no_glerror();

}

void DeferredLighting::renderLightAccumulation()
{
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    lightAccumulationShader->bind();
    lightAccumulationShader->uploadFramebuffer(&gbuffer);
    lightAccumulationShader->uploadLightAccumulationtexture(lightAccumulationTexture);
    lightAccumulationShader->uploadScreenSize(vec2(width,height));

    directionalLightMesh.bindAndDraw();

    lightAccumulationShader->unbind();

    glEnable(GL_DEPTH_TEST);
}

void DeferredLighting::setupStencilPass(){
    glEnable(GL_DEPTH_TEST);

    glColorMask(GL_FALSE,GL_FALSE,GL_FALSE,GL_FALSE);

    glClear(GL_STENCIL_BUFFER_BIT);

    glDisable(GL_CULL_FACE);

    // We need the stencil test to be enabled but we want it
    // to succeed always. Only the depth test matters.
    glStencilFunc(GL_ALWAYS, 0, 0);

    glStencilOpSeparate(GL_BACK, GL_KEEP, GL_DECR_WRAP, GL_KEEP);
    glStencilOpSeparate(GL_FRONT, GL_KEEP, GL_INCR_WRAP, GL_KEEP);


}
void DeferredLighting::setupLightPass(){
    // Disable color/depth write and enable stencil

    glStencilFunc(GL_NOTEQUAL, 0, 0xFF); //pass when pixel is inside a light volume
    glStencilOp( GL_KEEP, GL_KEEP, GL_KEEP);//do nothing
    glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);
    glDisable(GL_DEPTH_TEST);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_FRONT);
}



void DeferredLighting::renderDirectionalLights(Camera *cam,bool shadow){


    DirectionalLightShader* shader = (shadow)?directionalLightShadowShader:directionalLightShader;

    shader->bind();
    shader->uploadView(view);
    shader->uploadProj(proj);
    shader->DeferredShader::uploadFramebuffer(&gbuffer);
    shader->uploadScreenSize(vec2(width,height));
    shader->uploadSsaoTexture(ssaoTexture);

    directionalLightMesh.bind();
    for(DirectionalLight* &obj : directionalLights){
        bool render = (shadow&&obj->shouldCalculateShadowMap()) || (!shadow && obj->shouldRender() && !obj->hasShadows());
        if(render){
            obj->view = &view;
            obj->bindUniforms(*shader,cam);
            directionalLightMesh.draw();
        }
    }
    directionalLightMesh.unbind();
    shader->unbind();
}


void DeferredLighting::renderDebug(){

    debugShader->bind();
    debugShader->uploadView(view);
    debugShader->uploadProj(proj);

    // ======================= Pointlights ===================

    pointLightMesh.bind();
    //center
    for(PointLight* &obj : pointLights){
        mat4 sm = glm::scale(obj->model,vec3(0.05));
        vec4 color = obj->colorDiffuse;
        if(!obj->isActive()||!obj->isVisible()){
            //render as black if light is turned off
            color = vec4(0);
        }
        debugShader->uploadModel(sm);
        debugShader->uploadColor(color);
        pointLightMesh.draw();
    }

    //render outline
    glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
    for(PointLight* &obj : pointLights){
        //        if(obj->isSelected()){
        debugShader->uploadModel(obj->model);
        debugShader->uploadColor(obj->colorDiffuse);
        pointLightMesh.draw();
        //        }
    }
    pointLightMesh.unbind();
    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );


    //==================== Spotlights ==================

    spotLightMesh.bind();
    //center
    for(SpotLight* &obj : spotLights){
        mat4 sm = glm::scale(obj->model,vec3(0.05));
        vec4 color = obj->colorDiffuse;
        if(!obj->isActive()||!obj->isVisible()){
            //render as black if light is turned off
            color = vec4(0);
        }
        debugShader->uploadModel(sm);
        debugShader->uploadColor(color);
        spotLightMesh.draw();
    }

    //render outline
    glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
    for(SpotLight* &obj : spotLights){
        //        if(obj->isSelected()){
        debugShader->uploadModel(obj->model);
        debugShader->uploadColor(obj->colorDiffuse);
        spotLightMesh.draw();
        //        }
    }
    spotLightMesh.unbind();
    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );


    //==================== Box lights ====================

    boxLightMesh.bind();
    //center
    for(BoxLight* &obj : boxLights){
        mat4 sm = glm::scale(obj->model,vec3(0.05));
        vec4 color = obj->colorDiffuse;
        if(!obj->isActive()||!obj->isVisible()){
            //render as black if light is turned off
            color = vec4(0);
        }
        debugShader->uploadModel(sm);
        debugShader->uploadColor(color);
        boxLightMesh.draw();
    }

    //render outline
    glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
    for(BoxLight* &obj : boxLights){
        //        if(obj->isSelected()){
        debugShader->uploadModel(obj->model);
        debugShader->uploadColor(obj->colorDiffuse);
        boxLightMesh.draw();
        //        }
    }
    boxLightMesh.unbind();
    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );

    debugShader->unbind();

}

void DeferredLighting::blitGbufferDepthToAccumulationBuffer()
{
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_ALWAYS);
    blitDepthShader->bind();
    blitDepthShader->uploadTexture(gbuffer.getTextureDepth());
    directionalLightMesh.bindAndDraw();
    blitDepthShader->unbind();
    glDepthFunc(GL_LESS);
}

void DeferredLighting::setShader(SpotLightShader* spotLightShader, SpotLightShader* spotLightShadowShader){
    this->spotLightShader = spotLightShader;
    this->spotLightShadowShader = spotLightShadowShader;
}

void DeferredLighting::setShader(PointLightShader* pointLightShader, PointLightShader *pointLightShadowShader){
    this->pointLightShader = pointLightShader;
    this->pointLightShadowShader = pointLightShadowShader;
}

void DeferredLighting::setShader(DirectionalLightShader* directionalLightShader, DirectionalLightShader *directionalLightShadowShader){
    this->directionalLightShader = directionalLightShader;
    this->directionalLightShadowShader = directionalLightShadowShader;
}

void DeferredLighting::setShader(BoxLightShader *boxLightShader, BoxLightShader *boxLightShadowShader)
{
    this->boxLightShader = boxLightShader;
    this->boxLightShadowShader = boxLightShadowShader;
}

void DeferredLighting::setDebugShader(MVPColorShader *shader){
    this->debugShader = shader;
}

void DeferredLighting::setStencilShader(MVPShader* stencilShader){
    this->stencilShader = stencilShader;
}



void DeferredLighting::createInputCommands(){

}



void DeferredLighting::createLightMeshes(){

    auto qb = TriangleMeshGenerator::createFullScreenQuadMesh();
    qb->createBuffers(directionalLightMesh);

    Sphere s(vec3(0),1);
    auto sb = TriangleMeshGenerator::createMesh(s,1);
    sb->createBuffers(pointLightMesh);


    Cone c(vec3(0),vec3(0,1,0),30.0f,1.0f);
    auto cb = TriangleMeshGenerator::createMesh(c,10);
    cb->createBuffers(spotLightMesh);

    aabb box(vec3(-1),vec3(1));
    auto bb = TriangleMeshGenerator::createMesh(box);
    bb->createBuffers(boxLightMesh);
}

DirectionalLight* DeferredLighting::createDirectionalLight(){
    DirectionalLight* l = new DirectionalLight();
    directionalLights.push_back(l);
    return l;
}

PointLight* DeferredLighting::createPointLight(){
    PointLight* l = new PointLight();
    pointLights.push_back(l);
    return l;
}

SpotLight* DeferredLighting::createSpotLight(){
    SpotLight* l = new SpotLight();
    spotLights.push_back(l);
    return l;
}

BoxLight* DeferredLighting::createBoxLight(){
    BoxLight* l = new BoxLight();
    boxLights.push_back(l);
    return l;
}

void DeferredLighting::removeDirectionalLight(DirectionalLight *l)
{
    directionalLights.erase(std::find(directionalLights.begin(),directionalLights.end(),l));
}

void DeferredLighting::removePointLight(PointLight *l)
{
    pointLights.erase(std::find(pointLights.begin(),pointLights.end(),l));
}

void DeferredLighting::removeSpotLight(SpotLight *l)
{
    spotLights.erase(std::find(spotLights.begin(),spotLights.end(),l));
}

void DeferredLighting::removeBoxLight(BoxLight *l)
{
    boxLights.erase(std::find(boxLights.begin(),boxLights.end(),l));

}

void DeferredLighting::setViewProj(const mat4 &iv,const mat4 &v,const mat4 &p)
{
    inview = iv;
    view = v;
    proj = p;
}
