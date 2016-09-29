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

void DeferredLighting::init(int width, int height, bool _useTimers){
    this->width=width;this->height=height;
	useTimers = _useTimers;

	if (useTimers) {
		timers2.resize(5);
		for (int i = 0; i < 5; ++i) {
			timers2[i].create();

		}
		timerStrings.resize(5);
		timerStrings[0] = "Init";
		timerStrings[1] = "Point Lights";
		timerStrings[2] = "Spot Lights";
		timerStrings[3] = "Box Lights";
		timerStrings[4] = "Directional Lights";
	}

    lightAccumulationBuffer.create();

    //    Texture* depth_stencil = new Texture();
    //    depth_stencil->createEmptyTexture(width,height,GL_DEPTH_STENCIL, GL_DEPTH24_STENCIL8,GL_UNSIGNED_INT_24_8);
    //    lightAccumulationBuffer.attachTextureDepthStencil( framebuffer_texture_t(depth_stencil) );

    //NOTE: Use the same depth-stencil buffer as the gbuffer. I hope this works on every hardware :).
    lightAccumulationBuffer.attachTextureDepthStencil(gbuffer.getTextureDepth());

    lightAccumulationTexture = new Texture();
    lightAccumulationTexture->createEmptyTexture(width,height,GL_RGBA,GL_RGBA16,GL_UNSIGNED_SHORT);
    lightAccumulationBuffer.attachTexture( framebuffer_texture_t(lightAccumulationTexture) );
    lightAccumulationBuffer.drawToAll();
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

void DeferredLighting::printTimings()
{
	if (!useTimers)
		return;
    for(int i = 0 ;i < 5 ;++i){
        cout<<"\t "<< getTime(i)<<"ms "<<timerStrings[i]<<endl;
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

    startTimer(0);


    //viewport is maybe different after shadow map rendering
    glViewport(0,0,width,height);



    //    glClear( GL_STENCIL_BUFFER_BIT );
    //    glClear( GL_COLOR_BUFFER_BIT );

    //    glDepthMask(GL_FALSE);
    //    glDisable(GL_DEPTH_TEST);


    lightAccumulationBuffer.bind();

    blitGbufferDepthToAccumulationBuffer();






    //deferred lighting uses additive blending of the lights.
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_ONE, GL_ONE);

    //never overwrite current depthbuffer
    glDepthMask(GL_FALSE);

    //all light volumnes are using stencil culling
    glEnable(GL_STENCIL_TEST);

    //use depth test for all light volumnes
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);

    //    glClearStencil(0x0);
    //    glClear(GL_STENCIL_BUFFER_BIT);
    currentStencilId = 1;
    stopTimer(0);

    assert_no_glerror();
    startTimer(1);
    for(PointLight* l : pointLights){
        renderLightVolume<PointLight,PointLightShader>(pointLightMesh,l,cam,pointLightShader,pointLightShadowShader);
    }
    stopTimer(1);

    startTimer(2);
    for(SpotLight* l : spotLights){
        renderLightVolume<SpotLight,SpotLightShader>(spotLightMesh,l,cam,spotLightShader,spotLightShadowShader);
    }
    stopTimer(2);

    startTimer(3);
    for(BoxLight* l : boxLights){
        renderLightVolume<BoxLight,BoxLightShader>(boxLightMesh,l,cam,boxLightShader,boxLightShadowShader);
    }
    stopTimer(3);
    assert_no_glerror();

    //reset depth test to default value
    glDepthFunc(GL_LESS);




    //use default culling
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    glDisable(GL_DEPTH_TEST);

    startTimer(4);
    glStencilFunc(GL_NOTEQUAL, 0xFF, 0xFF);
    //    glStencilFunc(GL_EQUAL, 0x0, 0xFF);
    //    glStencilFunc(GL_ALWAYS, 0xFF, 0xFF);
    glStencilOp( GL_KEEP, GL_KEEP, GL_KEEP);
    renderDirectionalLights(cam,false);
    renderDirectionalLights(cam,true);
    stopTimer(4);

    glDisable(GL_STENCIL_TEST);
    assert_no_glerror();
    //reset state
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);

    if(drawDebug){
        glDepthMask(GL_TRUE);
        renderDebug();
        glDepthMask(GL_FALSE);
    }


    assert_no_glerror();

}



void DeferredLighting::setupStencilPass(){
    //don't write color in stencil pass
    glColorMask(GL_FALSE,GL_FALSE,GL_FALSE,GL_FALSE);

    //render only front faces
    glCullFace(GL_BACK);

    //default depth test
    glDepthFunc(GL_LEQUAL);

    //set stencil to 'id' if depth test fails
    //all 'failed' pixels are now marked in the stencil buffer with the id
    glStencilFunc(GL_ALWAYS, currentStencilId, 0xFF);
    //    glStencilFunc(GL_LEQUAL, currentStencilId, 0xFF);
    glStencilOp(GL_KEEP,GL_REPLACE,GL_KEEP);
}
void DeferredLighting::setupLightPass(){
    //write color in the light pass
    glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);

    //render only back faces
    glCullFace(GL_FRONT);

    //reversed depth test: it passes if the light volumne is behind an object
    glDepthFunc(GL_GEQUAL);

    //discard all pixels that are marked with 'id' from the previous pass
    glStencilFunc(GL_NOTEQUAL, currentStencilId, 0xFF);
    //    glStencilFunc(GL_NEVER, currentStencilId, 0xFF);
    //    glStencilFunc(GL_GREATER, currentStencilId, 0xFF);
    //    glStencilFunc(GL_EQUAL, 0x0, 0xFF);
    glStencilOp( GL_KEEP, GL_KEEP, GL_KEEP);

    //-> the reverse depth test + the stencil test make now sure that the current pixel is in the light volumne
    //this also works, when the camera is inside the volumne, but fails when the far plane is intersecting the volumne


    //increase stencil id, so the next light will write a different value to the stencil buffer.
    //with this trick the expensive clear can be saved after each light
    currentStencilId++;
    assert(currentStencilId<256);
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
            obj->bindUniforms(*shader,cam);
            directionalLightMesh.draw();
        }
    }
    directionalLightMesh.unbind();
    shader->unbind();
}


void DeferredLighting::renderDirectionalLight(DirectionalLight* obj, Camera *cam){
    if(!obj->shouldRender())
        return;


    DirectionalLightShader* shader = (obj->hasShadows()) ? directionalLightShadowShader : directionalLightShader;
    shader->bind();
    shader->uploadView(view);
    shader->uploadProj(proj);
    shader->DeferredShader::uploadFramebuffer(&gbuffer);
    shader->uploadScreenSize(vec2(width,height));
    shader->uploadSsaoTexture(ssaoTexture);
    obj->bindUniforms(*shader,cam);
    directionalLightMesh.bindAndDraw();
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
    //    glEnable(GL_DEPTH_TEST);
    //    glDepthFunc(GL_ALWAYS);
    //    blitDepthShader->bind();
    //    blitDepthShader->uploadTexture(gbuffer.getTextureDepth().get());
    //    directionalLightMesh.bindAndDraw();
    //    blitDepthShader->unbind();
    //    glDepthFunc(GL_LESS);




    //    glBindFramebuffer(GL_READ_FRAMEBUFFER, gbuffer.getId());
    //    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, lightAccumulationBuffer.getId());
    //    glBlitFramebuffer(0, 0, gbuffer.getTextureDepth()->getWidth(), gbuffer.getTextureDepth()->getHeight(), 0, 0, gbuffer.getTextureDepth()->getWidth(), gbuffer.getTextureDepth()->getHeight(),GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT, GL_NEAREST);

    glClearColor(0,0,0,0);
    glClear( GL_COLOR_BUFFER_BIT );
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
