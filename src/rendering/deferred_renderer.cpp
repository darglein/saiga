#include "rendering/deferred_renderer.h"

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


void Deferred_Renderer::init(DeferredShader* deferred_shader, int w, int h){
    setSize(w,h);
    lighting.setSize(w,h);
    deferred_framebuffer.create();
    deferred_framebuffer.makeToDeferredFramebuffer(w,h);

    mix_framebuffer.create();

    Texture* depth_stencil = new Texture();
    depth_stencil->createEmptyTexture(w,h,GL_DEPTH_STENCIL, GL_DEPTH24_STENCIL8,GL_UNSIGNED_INT_24_8);
    mix_framebuffer.attachTextureDepthStencil(depth_stencil);

    Texture* color = new Texture();
    color->createEmptyTexture(w,h,GL_RGB,GL_RGB8,GL_UNSIGNED_BYTE);
    mix_framebuffer.attachTexture(color);

    glDrawBuffer( GL_COLOR_ATTACHMENT0);

    mix_framebuffer.check();
    mix_framebuffer.unbind();

    setDeferredMixer(deferred_shader);


    auto qb = TriangleMeshGenerator::createFullScreenQuadMesh();
    qb->createBuffers(quadMesh);
}

void Deferred_Renderer::setDeferredMixer(DeferredShader* deferred_shader){
    this->deferred_shader = deferred_shader;
}

void Deferred_Renderer::render_intern(){

    glViewport(0,0,width,height);
    glClear( GL_COLOR_BUFFER_BIT );
    glClear(GL_DEPTH_BUFFER_BIT);


    renderGBuffer(*currentCamera);


    renderDepthMaps(*currentCamera);

    glDisable(GL_DEPTH_TEST);
    glViewport(0,0,width,height);

    //copy depth to lighting framebuffer. that is needed for stencil culling
    deferred_framebuffer.blitDepth(mix_framebuffer.id);

    mix_framebuffer.bind();
    glClear( GL_COLOR_BUFFER_BIT );
    renderLighting(*currentCamera);


    renderOverlay(*currentCamera);
    mix_framebuffer.unbind();


    if(postProcessing)
        postProcess();
    else
        mix_framebuffer.blitColor(0);
}

void Deferred_Renderer::renderGBuffer(Camera *cam){
    deferred_framebuffer.bind();
    glViewport(0,0,width,height);
    glClear( GL_COLOR_BUFFER_BIT );
    glClear(GL_DEPTH_BUFFER_BIT);
    glClear(GL_STENCIL_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);

    if(wireframe){
        glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
        glLineWidth(wireframeLineSize);
    }
    render(cam);
    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );



    deferred_framebuffer.unbind();

}

void Deferred_Renderer::renderDepthMaps(Camera *cam){
    lighting.renderDepthMaps(this);
    //    renderDepth(*currentCamera);
}

void Deferred_Renderer::renderLighting(Camera *cam){
    glDepthMask(GL_FALSE);
    lighting.setViewProj(cam->model,cam->view,cam->proj);
    lighting.render(cam);
    glDisable(GL_BLEND);
}

void Deferred_Renderer::postProcess(){


    postProcessingShader->bind();

    vec4 screenSize(width,height,1.0/width,1.0/height);
    postProcessingShader->uploadScreenSize(screenSize);
    postProcessingShader->uploadTexture(mix_framebuffer.colorBuffers[0]);
    quadMesh.bindAndDraw();
    postProcessingShader->unbind();
}
