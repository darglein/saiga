#include "rendering/lighting/deferred_lighting.h"
#include "util/inputcontroller.h"

#include "rendering/deferred_renderer.h"



DeferredLighting::DeferredLighting(Framebuffer &framebuffer):framebuffer(framebuffer){
    
    createInputCommands();
    createLightMeshes();




}

DeferredLighting::~DeferredLighting(){
    //delete all lights
    //the shader loader will delete the shaders.
    for(PointLight* &obj : pointLights){
        delete obj;
    }
    for(SpotLight* &obj : spotLights){
        delete obj;
    }
    for(DirectionalLight* &obj : directionalLights){
        delete obj;
    }
}

void DeferredLighting::renderDepthMaps(Deferred_Renderer *renderer){

    for(DirectionalLight* &light : directionalLights){

        if(light->hasShadows()){
            light->bindShadowMap();
            renderer->renderDepth(&light->cam);
            light->unbindShadowMap();
        }

    }

    for(SpotLight* &light : spotLights){
        if(light->hasShadows()){
            light->calculateCamera();
            light->bindShadowMap();
            renderer->renderDepth(&light->cam);
            light->unbindShadowMap();
        }
    }


    for(PointLight* &light : pointLights){

        if(light->hasShadows()){
            for(int i=0;i<6;i++){
                light->bindFace(i);
                light->calculateCamera(i);
                renderer->renderDepth(&light->cam);
                light->unbindShadowMap();
            }

        }
    }

}

void DeferredLighting::render(Camera* cam){

    glViewport(0,0,width,height);

    //============= Point lights


    renderSpotLightsStencil();


    renderSpotLights(cam);

    Error::quitWhenError("DeferredLighting::spotLights");


    renderPointLightsStencil();


    renderPointLights(cam);

    Error::quitWhenError("DeferredLighting::pointLights");


    //    glEnable(GL_DEPTH_TEST);
    //    glEnable(GL_STENCIL_TEST);
    //    glDepthMask(GL_FALSE);
    //============= Spot lights
    //    glClear(GL_STENCIL_BUFFER_BIT);
    //    renderSpotLightsStencil();
    //    glClear(GL_STENCIL_BUFFER_BIT);

    //    glStencilFunc(GL_NOTEQUAL, 0, 0xFF);
    //    glEnable(GL_BLEND);
    //    glBlendEquation(GL_FUNC_ADD);
    //    glBlendFunc(GL_ONE, GL_ONE);
    //    renderSpotLights();
    //    glDisable(GL_BLEND);




    renderDirectionalLights(cam);

    //draw solid on top
    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
    if(drawDebug)
        renderDebug();
    glDepthMask(GL_FALSE);


    Error::quitWhenError("DeferredLighting::lighting");

}

void DeferredLighting::setupStencilPass(){
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_STENCIL_TEST);

    glDepthMask(GL_FALSE);
    glColorMask(GL_FALSE,GL_FALSE,GL_FALSE,GL_FALSE);

    glClear(GL_STENCIL_BUFFER_BIT);

    glDisable(GL_CULL_FACE);

    // We need the stencil test to be enabled but we want it
    // to succeed always. Only the depth test matters.
    glStencilFunc(GL_ALWAYS, 0, 0);

    glStencilOpSeparate(GL_BACK, GL_KEEP, GL_INCR_WRAP, GL_KEEP);
    glStencilOpSeparate(GL_FRONT, GL_KEEP, GL_DECR_WRAP, GL_KEEP);


}
void DeferredLighting::setupLightPass(){
    // Disable color/depth write and enable stencil

    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_ONE, GL_ONE);


    glEnable(GL_STENCIL_TEST);
    glStencilFunc(GL_NOTEQUAL, 0, 0xFF);
    glStencilOp( GL_KEEP, GL_KEEP, GL_KEEP);//do nothing
    glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_FRONT);
}


void DeferredLighting::renderPointLights(Camera *cam){

    setupLightPass();
    pointLightShader->bind();
    pointLightShader->uploadView(view);
    pointLightShader->uploadProj(proj);
    pointLightShader->DeferredShader::uploadFramebuffer(&framebuffer);
    pointLightShader->uploadScreenSize(vec2(width,height));

//    Error::quitWhenError("DeferredLighting::renderPointLights1");

    pointLightMesh.bind();
    for(PointLight* &obj : pointLights){
        if(obj->isActive()&&obj->isVisible()){
            obj->bindUniforms(*pointLightShader,cam);
            pointLightMesh.draw();
        }
    }
    pointLightMesh.unbind();
    pointLightShader->unbind();

      Error::quitWhenError("DeferredLighting::renderPointLights4");
}

void DeferredLighting::renderPointLightsStencil(){

    setupStencilPass();
    stencilShader->bind();
    stencilShader->uploadView(view);
    stencilShader->uploadProj(proj);
    pointLightMesh.bind();
    for(PointLight* &obj : pointLights){
        if(obj->isActive()&&obj->isVisible()){

            obj->bindUniformsStencil(*stencilShader);
            pointLightMesh.draw();
        }
    }
    pointLightMesh.unbind();
    stencilShader->unbind();
}


void DeferredLighting::renderSpotLights(Camera *cam){


    setupLightPass();

    spotLightShader->bind();
    spotLightShader->uploadView(view);
    spotLightShader->uploadProj(proj);
    spotLightShader->DeferredShader::uploadFramebuffer(&framebuffer);
    spotLightShader->uploadScreenSize(vec2(width,height));

    spotLightMesh.bind();
    for(SpotLight* &obj : spotLights){
        if(obj->isActive()&&obj->isVisible()){
            obj->bindUniforms(*spotLightShader,cam);
            spotLightMesh.draw();
        }
    }
    spotLightMesh.unbind();
    spotLightShader->unbind();
}

void DeferredLighting::renderSpotLightsStencil(){
    setupStencilPass();

    stencilShader->bind();
    stencilShader->uploadView(view);
    stencilShader->uploadProj(proj);
    spotLightMesh.bind();
    for(SpotLight* &obj : spotLights){
        if(obj->isActive()&&obj->isVisible()){

            obj->bindUniformsStencil(*stencilShader);
            spotLightMesh.draw();
        }
    }
    spotLightMesh.unbind();
    stencilShader->unbind();
}


void DeferredLighting::renderDirectionalLights(Camera *cam){
    //reset stencil test
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_STENCIL_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    //    glEnable(GL_DEPTH_TEST);

    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);

    directionalLightShader->bind();
    directionalLightShader->uploadView(view);
    directionalLightShader->uploadProj(proj);
    directionalLightShader->DeferredShader::uploadFramebuffer(&framebuffer);
    directionalLightShader->uploadScreenSize(vec2(width,height));

    directionalLightMesh.bind();
    for(DirectionalLight* &obj : directionalLights){
        if(obj->isActive()&&obj->isVisible()){
            obj->view = &view;
            obj->bindUniforms(*directionalLightShader,cam);
            directionalLightMesh.draw();
        }
    }
    directionalLightMesh.unbind();
    directionalLightShader->unbind();
}


void DeferredLighting::renderDebug(){

    debugShader->bind();
    debugShader->uploadView(view);
    debugShader->uploadProj(proj);
    //    debugShader->uploadFramebuffer(&framebuffer);
    //    debugShader->uploadScreenSize(vec2(width,height));

    pointLightMesh.bind();
    //center
    for(PointLight* &obj : pointLights){
        mat4 small = glm::scale(obj->model,vec3(0.05));
        vec4 color = obj->color;
        if(!obj->isActive()||!obj->isVisible()){
            //render as black if light is turned off
            color = vec4(0);
        }
        debugShader->uploadModel(small);
        debugShader->uploadColor(color);
        pointLightMesh.draw();
    }

    //render outline
    glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
    for(PointLight* &obj : pointLights){
        if(obj->isSelected()){
            debugShader->uploadModel(obj->model);
            debugShader->uploadColor(obj->color);
            pointLightMesh.draw();
        }
    }
    pointLightMesh.unbind();
    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );


    spotLightMesh.bind();
    //center
    for(SpotLight* &obj : spotLights){
        mat4 small = glm::scale(obj->model,vec3(0.05));
        vec4 color = obj->color;
        if(!obj->isActive()||!obj->isVisible()){
            //render as black if light is turned off
            color = vec4(0);
        }
        debugShader->uploadModel(small);
        debugShader->uploadColor(color);
        spotLightMesh.draw();
    }

    //render outline
    glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
    for(SpotLight* &obj : spotLights){
        if(obj->isSelected()){
            debugShader->uploadModel(obj->model);
            debugShader->uploadColor(obj->color);
            spotLightMesh.draw();
        }
    }
    spotLightMesh.unbind();
    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );

    debugShader->unbind();

}


void DeferredLighting::setShader(SpotLightShader* spotLightShader){
    this->spotLightShader = spotLightShader;
}

void DeferredLighting::setShader(PointLightShader* pointLightShader){
    this->pointLightShader = pointLightShader;
}

void DeferredLighting::setShader(DirectionalLightShader* directionalLightShader){
    this->directionalLightShader = directionalLightShader;
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
}

DirectionalLight* DeferredLighting::createDirectionalLight(){
    directionalLights.push_back(new DirectionalLight());
    return directionalLights[directionalLights.size()-1];
}

PointLight* DeferredLighting::createPointLight(){
    pointLights.push_back(new PointLight());
    return pointLights[pointLights.size()-1];
}

SpotLight* DeferredLighting::createSpotLight(){
    spotLights.push_back(new SpotLight());
    return spotLights[spotLights.size()-1];
}

void DeferredLighting::setViewProj(const mat4 &iv,const mat4 &v,const mat4 &p)
{
    inview = iv;
    view = v;
    proj = p;
}
