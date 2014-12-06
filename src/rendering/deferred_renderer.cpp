#include "rendering/deferred_renderer.h"

void Deferred_Renderer::init(DeferredShader* deferred_shader, int w, int h){
    setSize(w,h);
    lighting.setSize(w,h);
    deferred_framebuffer.create();
    deferred_framebuffer.makeToDeferredFramebuffer(w,h);


    setDeferredMixer(deferred_shader);
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


    renderLighting(*currentCamera);


    renderOverlay(*currentCamera);

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
    deferred_framebuffer.blitDepth();
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
