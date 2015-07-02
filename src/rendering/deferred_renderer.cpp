#include "rendering/deferred_renderer.h"
#include "libhello/util/error.h"
#include "libhello/geometry/triangle_mesh_generator.h"
#include "libhello/camera/camera.h"


void SSAOShader::checkUniforms(){
    DeferredShader::checkUniforms();
    location_invProj = getUniformLocation("invProj");
    location_filterRadius = getUniformLocation("filterRadius");
    location_distanceThreshold = getUniformLocation("distanceThreshold");
}



void SSAOShader::uploadInvProj(mat4 &mat){
    Shader::upload(location_invProj,mat);
}

void SSAOShader::uploadData(){
    Shader::upload(location_filterRadius,filterRadius);
    Shader::upload(location_distanceThreshold,distanceThreshold);
}



Deferred_Renderer::Deferred_Renderer():lighting(deferred_framebuffer){

}

Deferred_Renderer::~Deferred_Renderer()
{

}

void Deferred_Renderer::init(DeferredShader* deferred_shader, int w, int h){
    setSize(w,h);
    lighting.init(w,h);
    deferred_framebuffer.create();
    deferred_framebuffer.makeToDeferredFramebuffer(w,h);


    ssao_framebuffer.create();
    Texture* ssaotex = new Texture();
    ssaotex->createEmptyTexture(w,h,GL_RED,GL_R8,GL_UNSIGNED_BYTE);
    ssao_framebuffer.attachTexture(ssaotex);
    glDrawBuffer( GL_COLOR_ATTACHMENT0);
    ssao_framebuffer.check();

    glClearColor(1.0f,1.0f,1.0f,1.0f);
    glClear( GL_COLOR_BUFFER_BIT );
    glClearColor(0.0f,0.0f,0.0f,0.0f);

    lighting.ssaoTexture = ssaotex;
    ssao_framebuffer.unbind();

    postProcessor.init(w,h);


//    initCudaPostProcessing(ppsrc,ppdst);


    setDeferredMixer(deferred_shader);


    auto qb = TriangleMeshGenerator::createFullScreenQuadMesh();
    qb->createBuffers(quadMesh);
}

void Deferred_Renderer::setDeferredMixer(DeferredShader* deferred_shader){
    this->deferred_shader = deferred_shader;
}

void Deferred_Renderer::toggleSSAO()
{
    ssao_framebuffer.bind();

    //clear with 1 -> no ambient occlusion
    glClearColor(1.0f,1.0f,1.0f,1.0f);
    glClear( GL_COLOR_BUFFER_BIT );
    glClearColor(0.0f,0.0f,0.0f,0.0f);
    ssao_framebuffer.unbind();

    ssao = !ssao;
}

void Deferred_Renderer::render_intern(){


    (*currentCamera)->recalculatePlanes();

    renderGBuffer(*currentCamera);

    renderSSAO(*currentCamera);

    lighting.cullLights(*currentCamera);
    renderDepthMaps(*currentCamera);

    glDisable(GL_DEPTH_TEST);
    glViewport(0,0,width,height);

    Error::quitWhenError("Deferred_Renderer::before blit");

    //copy depth to lighting framebuffer. that is needed for stencil culling


    Error::quitWhenError("Deferred_Renderer::after blit");


    //    mix_framebuffer.bind();
    //    glClear( GL_COLOR_BUFFER_BIT );


    renderLighting(*currentCamera);


    postProcessor.nextFrame(&deferred_framebuffer);

    postProcessor.bindCurrentBuffer();
    lighting.renderLightAccumulation();


    renderer->renderOverlay(*currentCamera);

    postProcessor.switchBuffer();

    postProcessor.render();




    renderer->renderFinal(*currentCamera);



    Error::quitWhenError("Deferred_Renderer::render_intern");

}

void Deferred_Renderer::renderGBuffer(Camera *cam){
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);


    deferred_framebuffer.bind();
    glViewport(0,0,width,height);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);



    if(wireframe){
        glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
        glLineWidth(wireframeLineSize);
    }
    renderer->render(cam);
    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );



    deferred_framebuffer.unbind();


    Error::quitWhenError("Deferred_Renderer::renderGBuffer");

}

void Deferred_Renderer::renderDepthMaps(Camera *cam){

    // When GL_POLYGON_OFFSET_FILL, GL_POLYGON_OFFSET_LINE, or GL_POLYGON_OFFSET_POINT is enabled,
    // each fragment's depth value will be offset after it is interpolated from the depth values of the appropriate vertices.
    // The value of the offset is factor×DZ+r×units, where DZ is a measurement of the change in depth relative to the screen area of the polygon,
    // and r is the smallest value that is guaranteed to produce a resolvable offset for a given implementation.
    // The offset is added before the depth test is performed and before the value is written into the depth buffer.
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1.0f,4.0f);
    lighting.renderDepthMaps(renderer);
    glDisable(GL_POLYGON_OFFSET_FILL);

    Error::quitWhenError("Deferred_Renderer::renderDepthMaps");

}

void Deferred_Renderer::renderLighting(Camera *cam){
    mat4 model;
    cam->getModelMatrix(model);
    lighting.setViewProj(model,cam->view,cam->proj);
    lighting.render(cam);
    Error::quitWhenError("Deferred_Renderer::renderLighting");
}

void Deferred_Renderer::renderSSAO(Camera *cam)
{

    if(!ssao)
        return;

    ssao_framebuffer.bind();

    //    glClearColor(1.0f,1.0f,1.0f,1.0f);
    //    glClear( GL_COLOR_BUFFER_BIT );


    if(ssaoShader){
        ssaoShader->bind();
        vec2 screenSize(width,height);
        ssaoShader->uploadScreenSize(screenSize);
        ssaoShader->uploadFramebuffer(&deferred_framebuffer);
        ssaoShader->uploadData();
        mat4 iproj = glm::inverse(cam->proj);
        ssaoShader->uploadInvProj(iproj);
        quadMesh.bindAndDraw();
        ssaoShader->unbind();
    }


    ssao_framebuffer.unbind();

}

