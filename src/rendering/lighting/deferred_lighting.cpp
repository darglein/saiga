#include "rendering/lighting/deferred_lighting.h"
#include "util/inputcontroller.h"

#include "rendering/deferred_renderer.h"
#include "libhello/util/error.h"

#include "libhello/rendering/lighting/directional_light.h"
#include "libhello/rendering/lighting/point_light.h"
#include "libhello/rendering/lighting/spot_light.h"
#include "libhello/rendering/lighting/box_light.h"

#include "libhello/geometry/triangle_mesh_generator.h"
#include "libhello/opengl/texture/cube_texture.h"

DeferredLighting::DeferredLighting(Framebuffer &framebuffer):framebuffer(framebuffer){
    
    createInputCommands();
    createLightMeshes();



    //    dummyTexture = new Texture();
    //    dummyTexture->createEmptyTexture(1,1,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT16,GL_UNSIGNED_SHORT);
    //    dummyTexture->setWrap(GL_CLAMP_TO_EDGE);
    //    dummyTexture->setFiltering(GL_LINEAR);
    //    //this requires the texture sampler in the shader to be sampler2DShadow
    //    dummyTexture->setParameter(GL_TEXTURE_COMPARE_MODE,GL_COMPARE_REF_TO_TEXTURE);
    //    dummyTexture->setParameter(GL_TEXTURE_COMPARE_FUNC,GL_LEQUAL);



    //    dummyCubeTexture = new cube_Texture();
    //    dummyCubeTexture->createEmptyTexture(1,1,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT16,GL_UNSIGNED_SHORT);
    //    dummyCubeTexture->setWrap(GL_CLAMP_TO_EDGE);
    //    dummyCubeTexture->setFiltering(GL_LINEAR);
    //    //this requires the texture sampler in the shader to be sampler2DShadow
    //    dummyCubeTexture->setParameter(GL_TEXTURE_COMPARE_MODE,GL_COMPARE_REF_TO_TEXTURE);
    //    dummyCubeTexture->setParameter(GL_TEXTURE_COMPARE_FUNC,GL_LEQUAL);
}

DeferredLighting::~DeferredLighting(){
    //    delete dummyTexture;
    //    delete dummyCubeTexture;
    //delete all lights
    //the shader loader will delete the shaders.
    //    for(PointLight* &obj : pointLights){
    //        delete obj;
    //    }
    //    for(SpotLight* &obj : spotLights){
    //        delete obj;
    //    }
    //    for(DirectionalLight* &obj : directionalLights){
    //        delete obj;
    //    }
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

void DeferredLighting::renderDepthMaps(RendererInterface *renderer){
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
    //viewport is maybe different after shadow map rendering
    glViewport(0,0,width,height);

    //deferred lighting uses additive blending of the lights.
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_ONE, GL_ONE);

    //never overwrite current depthbuffer
    glDepthMask(GL_FALSE);

    //point- and spot- lights are using stencil culling
    glEnable(GL_STENCIL_TEST);


    renderSpotLightsStencil(); //mark pixels inside the light volume
    setupLightPass();
    renderSpotLights(cam,false); //draw back faces without depthtest
    renderSpotLights(cam,true);

    Error::quitWhenError("DeferredLighting::spotLights");

    renderPointLightsStencil(); //mark pixels inside the light volume
    setupLightPass();
    renderPointLights(cam,false); //draw back faces without depthtest
    renderPointLights(cam,true);


    Error::quitWhenError("DeferredLighting::pointLights");

    renderBoxLightsStencil(); //mark pixels inside the light volume
    setupLightPass();
    renderBoxLights(cam,false); //draw back faces without depthtest
    renderBoxLights(cam,true);


    Error::quitWhenError("DeferredLighting::boxLights");

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



    Error::quitWhenError("DeferredLighting::lighting");

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


void DeferredLighting::renderPointLights(Camera *cam, bool shadow){

    PointLightShader* shader = (shadow)?pointLightShadowShader:pointLightShader;


    shader->bind();
    shader->uploadView(view);
    shader->uploadProj(proj);
    shader->DeferredShader::uploadFramebuffer(&framebuffer);
    shader->uploadScreenSize(vec2(width,height));

    //    Error::quitWhenError("DeferredLighting::renderPointLights1");

    pointLightMesh.bind();
    for(PointLight* &obj : pointLights){

        bool render = (shadow&&obj->shouldCalculateShadowMap()) || (!shadow && obj->shouldRender() && !obj->hasShadows());        if(render){
            obj->bindUniforms(*shader,cam);
            pointLightMesh.draw();
        }
    }
    pointLightMesh.unbind();
    shader->unbind();

    Error::quitWhenError("DeferredLighting::renderPointLights4");
}

void DeferredLighting::renderPointLightsStencil(){

    setupStencilPass();
    stencilShader->bind();
    stencilShader->uploadView(view);
    stencilShader->uploadProj(proj);
    pointLightMesh.bind();
    for(PointLight* &obj : pointLights){
        if(obj->shouldRender()){

            obj->bindUniformsStencil(*stencilShader);
            pointLightMesh.draw();
        }
    }
    pointLightMesh.unbind();
    stencilShader->unbind();
}


void DeferredLighting::renderSpotLights(Camera *cam, bool shadow){

    SpotLightShader* shader = (shadow)?spotLightShadowShader:spotLightShader;

    shader->bind();
    shader->uploadView(view);
    shader->uploadProj(proj);
    shader->DeferredShader::uploadFramebuffer(&framebuffer);
    shader->uploadScreenSize(vec2(width,height));

    spotLightMesh.bind();
    for(SpotLight* &obj : spotLights){
        bool render = (shadow&&obj->shouldCalculateShadowMap()) || (!shadow && obj->shouldRender() && !obj->hasShadows());
        if(render){
            obj->bindUniforms(*shader,cam);
            spotLightMesh.draw();
        }
    }
    spotLightMesh.unbind();
    shader->unbind();

}

void DeferredLighting::renderSpotLightsStencil(){
    setupStencilPass();

    stencilShader->bind();
    stencilShader->uploadView(view);
    stencilShader->uploadProj(proj);
    spotLightMesh.bind();
    for(SpotLight* &obj : spotLights){
        if(obj->shouldRender()){

            obj->bindUniformsStencil(*stencilShader);
            spotLightMesh.draw();
        }
    }
    spotLightMesh.unbind();
    stencilShader->unbind();
}



void DeferredLighting::renderBoxLights(Camera *cam, bool shadow){

    BoxLightShader* shader = (shadow)?boxLightShadowShader:boxLightShader;

    shader->bind();
    shader->uploadView(view);
    shader->uploadProj(proj);
    shader->DeferredShader::uploadFramebuffer(&framebuffer);
    shader->uploadScreenSize(vec2(width,height));

    boxLightMesh.bind();
    for(BoxLight* &obj : boxLights){
        bool render = (shadow&&obj->shouldCalculateShadowMap()) || (!shadow && obj->shouldRender() && !obj->hasShadows());
        if(render){
            obj->view = &view;
            obj->bindUniforms(*shader,cam);
            boxLightMesh.draw();
        }
    }
    boxLightMesh.unbind();
    shader->unbind();

}

void DeferredLighting::renderBoxLightsStencil(){
    setupStencilPass();

    stencilShader->bind();
    stencilShader->uploadView(view);
    stencilShader->uploadProj(proj);
    boxLightMesh.bind();
    for(BoxLight* &obj : boxLights){
        if(obj->shouldRender()){

            obj->bindUniformsStencil(*stencilShader);
            boxLightMesh.draw();
        }
    }
    boxLightMesh.unbind();
    stencilShader->unbind();
}

void DeferredLighting::renderDirectionalLights(Camera *cam,bool shadow){


    DirectionalLightShader* shader = (shadow)?directionalLightShadowShader:directionalLightShader;

    shader->bind();
    shader->uploadView(view);
    shader->uploadProj(proj);
    shader->DeferredShader::uploadFramebuffer(&framebuffer);
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
        vec4 color = obj->color;
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
        debugShader->uploadColor(obj->color);
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
        vec4 color = obj->color;
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
        debugShader->uploadColor(obj->color);
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
        vec4 color = obj->color;
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
        debugShader->uploadColor(obj->color);
        boxLightMesh.draw();
        //        }
    }
    boxLightMesh.unbind();
    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );

    debugShader->unbind();

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
