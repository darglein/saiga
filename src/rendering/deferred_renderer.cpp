#include "rendering/deferred_renderer.h"
#include "libhello/util/error.h"
void PostProcessingShader::checkUniforms(){
    Shader::checkUniforms();
    location_texture = Shader::getUniformLocation("image");
    location_screenSize = Shader::getUniformLocation("screenSize");
}


void PostProcessingShader::uploadTexture(raw_Texture *texture){
    texture->bind(0);
    Shader::upload(location_texture,0);
}

void PostProcessingShader::uploadScreenSize(vec4 size){
    Shader::upload(location_screenSize,size);
}



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
    lighting.setSize(w,h);
    deferred_framebuffer.create();
    deferred_framebuffer.makeToDeferredFramebuffer(w,h);

    mix_framebuffer.create();
    Texture* depth_stencil = new Texture();
        depth_stencil->createEmptyTexture(w,h,GL_DEPTH_STENCIL, GL_DEPTH24_STENCIL8,GL_UNSIGNED_INT_24_8);
    mix_framebuffer.attachTextureDepthStencil(depth_stencil);

    //different textures for depth and stencil not supported!
    //However, implementations are only required to support both a depth and stencil attachment simultaneously if both attachments refer to the same image.
//    depth_stencil->createEmptyTexture(w,h,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT32,GL_UNSIGNED_INT);
//    mix_framebuffer.attachTextureDepth(depth_stencil);
//    Texture* stencil = new Texture();
//    stencil->createEmptyTexture(w,h, GL_STENCIL_INDEX, GL_STENCIL_INDEX8,GL_UNSIGNED_BYTE);
//     mix_framebuffer.attachTextureStencil(stencil);


    Texture* ppsrc = new Texture();
    //    ppsrc->createEmptyTexture(w,h,GL_RGBA,GL_SRGB8_ALPHA8,GL_UNSIGNED_BYTE);
    //    ppsrc->createEmptyTexture(w,h,GL_RGBA,GL_RGBA8,GL_UNSIGNED_BYTE);
    ppsrc->createEmptyTexture(w,h,GL_RGB,GL_RGB16,GL_UNSIGNED_SHORT);
    //     ppsrc->createEmptyTexture(w,h,GL_RGBA,GL_RGBA32F,GL_FLOAT);
    mix_framebuffer.attachTexture(ppsrc);
    glDrawBuffer( GL_COLOR_ATTACHMENT0);
    mix_framebuffer.check();
    mix_framebuffer.unbind();


    postProcess_framebuffer.create();
    Texture* ppdst = new Texture();
    ppdst->createEmptyTexture(w,h,GL_RGBA,GL_RGBA8,GL_UNSIGNED_BYTE);
    //     ppdst->createEmptyTexture(w,h,GL_RGBA,GL_SRGB8_ALPHA8,GL_UNSIGNED_BYTE);
    postProcess_framebuffer.attachTexture(ppdst);
    glDrawBuffer( GL_COLOR_ATTACHMENT0);
    postProcess_framebuffer.check();
    postProcess_framebuffer.unbind();


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


    initCudaPostProcessing(ppsrc,ppdst);


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

    cout<<"Deferred_Renderer::render_intern"<<endl;
    //    glViewport(0,0,width,height);
    //    glClear( GL_COLOR_BUFFER_BIT );
    //    glClear(GL_DEPTH_BUFFER_BIT);


    (*currentCamera)->recalculatePlanes();

    renderGBuffer(*currentCamera);

    renderSSAO(*currentCamera);

    lighting.cullLights(*currentCamera);
    renderDepthMaps(*currentCamera);

    glDisable(GL_DEPTH_TEST);
    glViewport(0,0,width,height);

    Error::quitWhenError("Deferred_Renderer::before blit");


    //remove maybe
//    mix_framebuffer.bind();
//    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);


    //copy depth to lighting framebuffer. that is needed for stencil culling
    deferred_framebuffer.blitDepth(mix_framebuffer.id);

    Error::quitWhenError("Deferred_Renderer::after blit");

    //    glEnable(GL_FRAMEBUFFER_SRGB);

    mix_framebuffer.bind();
    glClear( GL_COLOR_BUFFER_BIT );



    renderLighting(*currentCamera);


    renderer->renderOverlay(*currentCamera);
    mix_framebuffer.unbind();

    //    glDisable(GL_FRAMEBUFFER_SRGB);

    if(postProcessing)
        postProcess();
    else{
        //        postProcess();
        mix_framebuffer.blitColor(0);
    }

    renderer->renderFinal(*currentCamera);



    Error::quitWhenError("Deferred_Renderer::render_intern");

}

void Deferred_Renderer::renderGBuffer(Camera *cam){
    deferred_framebuffer.bind();
    glViewport(0,0,width,height);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    glDisable(GL_BLEND);
//    glClear( GL_DEPTH_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);

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

    //    glClearColor(0.0f,0.0f,0.0f,0.0f);
}

void Deferred_Renderer::postProcess(){
    glDisable(GL_DEPTH_TEST);

    //remove maybe
    glDisable(GL_BLEND);

    //shader post process + gamma correction
    glEnable(GL_FRAMEBUFFER_SRGB);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    //    postProcess_framebuffer.bind();
    //    glClear( GL_COLOR_BUFFER_BIT );
    postProcessingShader->bind();

    vec4 screenSize(width,height,1.0/width,1.0/height);
    postProcessingShader->uploadScreenSize(screenSize);
    postProcessingShader->uploadTexture(mix_framebuffer.colorBuffers[0]);
    //    postProcessingShader->uploadTexture(ssao_framebuffer.colorBuffers[0]);
    postProcessingShader->uploadAdditionalUniforms();
    quadMesh.bindAndDraw();
    postProcessingShader->unbind();

    //    postProcess_framebuffer.unbind();

    glDisable(GL_FRAMEBUFFER_SRGB);

    //    postProcess_framebuffer.blitColor(0);

    //    cudaPostProcessing();
    //    postProcess_framebuffer.blitColor(0);

    //     Error::quitWhenError("Deferred_Renderer::postProcess");
}

