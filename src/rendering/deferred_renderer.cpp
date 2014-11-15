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
    mat4 &inview = (**currentCamera).model;
    mat4 &view = (**currentCamera).view;
    mat4 &proj  = (**currentCamera).proj;

    glViewport(0,0,width,height);
    glClear( GL_COLOR_BUFFER_BIT );
    glClear(GL_DEPTH_BUFFER_BIT);


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
    render();
    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
    glDisable(GL_DEPTH_TEST);


    deferred_framebuffer.unbind();




    deferred_framebuffer.blitDepth();

    //    glDisable(GL_CULL_FACE);
    //    renderLighting();
    //    glEnable(GL_CULL_FACE);

    //    glEnable(GL_BLEND);
    //        glBlendEquation(GL_FUNC_ADD);
    //        glBlendFunc(GL_ONE, GL_ONE);
    glDepthMask(GL_FALSE);
    lighting.setViewProj(inview,view,proj);
    lighting.render();
    glDisable(GL_BLEND);
    //    glDepthMask(GL_TRUE);

    //    glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
    //    glLineWidth(wireframeLineSize);
    //    lighting.renderDebug(view,proj,&deferred_framebuffer);




    renderOverlay();



}
