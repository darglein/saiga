/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/rendering/deferredRendering/deferredRendering.h"
#include "saiga/opengl/error.h"
#include "saiga/geometry/triangle_mesh_generator.h"
#include "saiga/opengl/shader/shaderLoader.h"

#include "saiga/camera/camera.h"
#include "saiga/rendering/renderer.h"
#include "saiga/window/window.h"
#include "saiga/imgui/imgui.h"
#include "saiga/rendering/program.h"

namespace Saiga {

Deferred_Renderer::Deferred_Renderer(OpenGLWindow &window, DeferredRenderingParameters _params) :
    Renderer(window),
    renderWidth(window.getWidth()*_params.renderScale), renderHeight(window.getHeight()*_params.renderScale),
    params(_params),lighting(gbuffer),
    ddo(window.getWidth(),window.getHeight())
{




    //    setSize(windowWidth,windowHeight);

    if(params.useSMAA){
        smaa = std::make_shared<SMAA>(renderWidth, renderHeight);
        smaa->loadShader(params.smaaQuality);
    }

    {
        //create a 2x2 grayscale black dummy texture
        blackDummyTexture = std::make_shared<Texture>();
        std::vector<int> data(2*2,0);
        blackDummyTexture->createTexture(2,2,GL_RED,GL_R8,GL_UNSIGNED_BYTE, (GLubyte*)data.data());

    }
    if(params.useSSAO){
        ssao = std::make_shared<SSAO>(renderWidth, renderHeight);
    }
    lighting.ssaoTexture = ssao ? ssao->bluredTexture : blackDummyTexture;
    //        ssao.init(windowWidth*params.renderScale, windowHeight*params.renderScale);
    //    else
    //        ssao.init(2,2);

    if(params.srgbWrites){

        //intel graphics drivers on windows do not define this extension but srgb still works..
        //SAIGA_ASSERT(hasExtension("GL_EXT_framebuffer_sRGB"));

        //Mesa drivers do not respect the spec when blitting with srgb framebuffers.
        //https://lists.freedesktop.org/archives/mesa-dev/2015-February/077681.html

        //TODO check for mesa
        //If this is true some recording softwares record the image too dark :(
        params.blitLastFramebuffer = false;
    }



    gbuffer.init(renderWidth, renderHeight, params.gbp);

    lighting.init(renderWidth, renderHeight, params.useGPUTimers);
    lighting.shadowSamples = params.shadowSamples;
    lighting.clearColor = params.lightingClearColor;
    lighting.loadShaders();



    postProcessor.init(renderWidth, renderHeight, &gbuffer, params.ppp, lighting.lightAccumulationTexture, params.useGPUTimers);


    auto qb = TriangleMeshGenerator::createFullScreenQuadMesh();
    qb->createBuffers(quadMesh);

    int numTimers = DeferredTimings::COUNT;
    if (!params.useGPUTimers)
        numTimers = 1; //still use one rendering timer :)
    timers.resize(numTimers);
    for (auto &t : timers) {
        t.create();
    }



    blitDepthShader = ShaderLoader::instance()->load<MVPTextureShader>("lighting/blitDepth.glsl");

    ddo.setDeferredFramebuffer(&gbuffer,lighting.volumetricLightTexture2);





    std::shared_ptr<PostProcessingShader>  pps = ShaderLoader::instance()->load<PostProcessingShader>("post_processing/post_processing.glsl"); //this shader does nothing
    std::vector<std::shared_ptr<PostProcessingShader> > defaultEffects;
    defaultEffects.push_back(pps);
    postProcessor.setPostProcessingEffects(defaultEffects);


    cout << "Deferred Renderer initialized. Render resolution: " << renderWidth << "x" << renderHeight << endl;

}

Deferred_Renderer::~Deferred_Renderer()
{

}



void Deferred_Renderer::resize(int windowWidth, int windowHeight)
{


    if (windowWidth <= 0 || windowHeight <= 0) {
        cerr << "Warning: The window size must be greater than zero." << endl;
        windowWidth = glm::max(windowWidth, 1);
        windowHeight = glm::max(windowHeight, 1);
    }
    this->outputWidth = windowWidth;
    this->outputHeight = windowHeight;
    this->renderWidth = windowWidth * params.renderScale;
    this->renderHeight = windowHeight * params.renderScale;
    cout << "Resizing Window to : " << windowWidth << "," << windowHeight << endl;
    cout << "Framebuffer size: " << renderWidth << " " << renderHeight << endl;
    postProcessor.resize(renderWidth, renderHeight);
    gbuffer.resize(renderWidth, renderHeight);
    lighting.resize(renderWidth, renderHeight);

    if(ssao)
        ssao->resize(renderWidth, renderHeight);

    if(smaa){
        smaa->resize(renderWidth,renderHeight);
    }
}






void Deferred_Renderer::render(Camera *cam)
{
    if(!rendering)
        return;

    SAIGA_ASSERT(rendering);
    SAIGA_ASSERT(cam);


    if (params.srgbWrites)
        glEnable(GL_FRAMEBUFFER_SRGB);

    startTimer(TOTAL);

    // When GL_FRAMEBUFFER_SRGB is disabled, the system assumes that the color written by the fragment shader
    // is in whatever colorspace the image it is being written to is. Therefore, no colorspace correction is performed.
    // If GL_FRAMEBUFFER_SRGB is enabled however, then if the destination image is in the sRGB colorspace
    // (as queried through glGetFramebufferAttachmentParameter(GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING)​),
    // then it will assume the shader's output is in the linear RGB colorspace.
    // It will therefore convert the output from linear RGB to sRGB.
    //    if (params.srgbWrites)
    //        glEnable(GL_FRAMEBUFFER_SRGB); //no reason to switch it off

    cam->recalculatePlanes();
    bindCamera(cam);
    renderGBuffer(cam);
    assert_no_glerror();


    renderSSAO(cam);
    //    return;

    lighting.initRender();
    lighting.cullLights(cam);
    renderDepthMaps();


    bindCamera(cam);
    renderLighting(cam);



    if (params.writeDepthToOverlayBuffer) {
        //        writeGbufferDepthToCurrentFramebuffer();
    }
    else {
        glClear(GL_DEPTH_BUFFER_BIT);
    }

    startTimer(OVERLAY);

    bindCamera(cam);
    rendering->renderOverlay(cam);
    stopTimer(OVERLAY);



    lighting.applyVolumetricLightBuffer();

    postProcessor.nextFrame();
    postProcessor.bindCurrentBuffer();
    //    postProcessor.switchBuffer();


    startTimer(POSTPROCESSING);
    //postprocessor's 'currentbuffer' will still be bound after render
    postProcessor.render();
    stopTimer(POSTPROCESSING);

    //    deferred_framebuffer.blitDepth(0);




    if(params.useSMAA){
        startTimer(SMAATIME);
        smaa->render(postProcessor.getCurrentTexture(),postProcessor.getTargetBuffer());
        postProcessor.switchBuffer();
        postProcessor.bindCurrentBuffer();
        stopTimer(SMAATIME);
    }

    //write depth to default framebuffer
    if (params.writeDepthToDefaultFramebuffer) {
        //        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        postProcessor.bindCurrentBuffer();
        writeGbufferDepthToCurrentFramebuffer();
    }


    //glBindFramebuffer(GL_FRAMEBUFFER, 0);
    //    glClear(GL_COLOR_BUFFER_BIT);
    startTimer(FINAL);
    glViewport(0, 0, renderWidth, renderHeight);
    if(renderDDO)
    {
        bindCamera(&ddo.layout.cam);
        ddo.render();
    }

    {
        //final render pass
        if(imgui)
        {
            imgui->beginFrame();
        }
        rendering->renderFinal(cam);
        if(imgui)
        {
            imgui->endFrame();
        }
    }
    stopTimer(FINAL);

    glDisable(GL_BLEND);

    if(params.blitLastFramebuffer)
        postProcessor.blitLast(outputWidth, outputHeight);
    else
        postProcessor.renderLast(outputWidth, outputHeight);

    //    if (params.srgbWrites)
    //        glDisable(GL_FRAMEBUFFER_SRGB);

    if (params.useGlFinish)
        glFinish();

    stopTimer(TOTAL);



    //    std::cout<<"Time spent on the GPU: "<< getTime(TOTAL) <<std::endl;

    //    printTimings();


    //    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);   // Make sure no FBO is set as the draw framebuffer
    //     glBindFramebuffer(GL_READ_FRAMEBUFFER, lighting.lightAccumulationBuffer.getId()); // Make sure your multisampled FBO is the read framebuffer
    //     glDrawBuffer(GL_BACK);                       // Set the back buffer as the draw buffer
    //     glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST);

    assert_no_glerror();

}

void Deferred_Renderer::renderGBuffer(Camera *cam) {
    startTimer(GEOMETRYPASS);

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);


    gbuffer.bind();
    glViewport(0, 0, renderWidth, renderHeight);
    glClearColor(params.clearColor.x, params.clearColor.y, params.clearColor.z, params.clearColor.w);

    if (params.maskUsedPixels) {
        glClearStencil(0xFF); //sets stencil buffer to 255
        //mark all written pixels with 0 in the stencil buffer
        glEnable(GL_STENCIL_TEST);
        glStencilFunc(GL_ALWAYS, 0, 0xFF);
        glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
    }
    else {
        glClearStencil(0x00);
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);


    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);


    if (params.offsetGeometry) {
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(params.offsetFactor, params.offsetUnits);
    }

    if (params.wireframe) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glLineWidth(params.wireframeLineSize);
    }
    rendering->render(cam);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);


    if (params.offsetGeometry) {
        glDisable(GL_POLYGON_OFFSET_FILL);
    }

    glDisable(GL_STENCIL_TEST);

    gbuffer.unbind();


    stopTimer(GEOMETRYPASS);

    assert_no_glerror();

}

void Deferred_Renderer::renderDepthMaps() {

    startTimer(DEPTHMAPS);


    lighting.renderDepthMaps(rendering);


    stopTimer(DEPTHMAPS);

    assert_no_glerror();

}

void Deferred_Renderer::renderLighting(Camera *cam) {
    startTimer(LIGHTING);

    lighting.render(cam);

    stopTimer(LIGHTING);

    assert_no_glerror();
}

void Deferred_Renderer::renderSSAO(Camera *cam)
{

    startTimer(SSAOT);

    if(params.useSSAO)
        ssao->render(cam, &gbuffer);


    stopTimer(SSAOT);

    assert_no_glerror();

}

void Deferred_Renderer::writeGbufferDepthToCurrentFramebuffer()
{
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDepthFunc(GL_ALWAYS);
    blitDepthShader->bind();
    blitDepthShader->uploadTexture(gbuffer.getTextureDepth());
    quadMesh.bindAndDraw();
    blitDepthShader->unbind();
    glDepthFunc(GL_LESS);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

    assert_no_glerror();
}



void Deferred_Renderer::printTimings()
{
    cout << "====================================" << endl;
    cout << "Geometry pass: " << getTime(GEOMETRYPASS) << "ms" << endl;
    cout << "SSAO: " << getTime(SSAOT) << "ms" << endl;
    cout << "Depthmaps: " << getTime(DEPTHMAPS) << "ms" << endl;
    cout << "Lighting: " << getTime(LIGHTING) << "ms" << endl;
    lighting.printTimings();
    //    cout<<"Light accumulation: "<<getTime(LIGHTACCUMULATION)<<"ms"<<endl;
    cout << "Overlay pass: " << getTime(OVERLAY) << "ms" << endl;
    cout << "Postprocessing: " << getTime(POSTPROCESSING) << "ms" << endl;
    postProcessor.printTimings();
    cout << "SMAA: " << getTime(SMAATIME) << "ms" << endl;
    cout << "Final pass: " << getTime(FINAL) << "ms" << endl;
    float total = getTime(TOTAL);
    cout << "Total: " << total << "ms (" << 1000 / total << " fps)" << endl;
    cout << "====================================" << endl;

}


void Deferred_Renderer::renderImGui(bool *p_open)
{
    int w = 340;
    int h = 240;
    ImGui::SetNextWindowPos(ImVec2(340, outputHeight - h), ImGuiSetCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(w,h), ImGuiSetCond_FirstUseEver);
    ImGui::Begin("Deferred Renderer",p_open);

    ImGui::Checkbox("renderDDO", &renderDDO);
    ImGui::Checkbox("wireframe", &params.wireframe);
    ImGui::Checkbox("offsetGeometry", &params.offsetGeometry);

    ImGui::Text("Render Time");
    ImGui::Text("%fms - Geometry pass",getTime(GEOMETRYPASS));
    ImGui::Text("%fms - SSAO",getTime(SSAOT));
    ImGui::Text("%fms - Depthmaps",getTime(DEPTHMAPS));
    ImGui::Text("%fms - Lighting",getTime(LIGHTING));
    ImGui::Text("%fms - Overlay pass",getTime(OVERLAY));
    ImGui::Text("%fms - Postprocessing",getTime(POSTPROCESSING));
    ImGui::Text("%fms - SMAA",getTime(SMAATIME));
    ImGui::Text("%fms - Final pass",getTime(FINAL));
    ImGui::Text("%fms - Total",getTime(TOTAL));

    ImGui::Separator();

    if(ImGui::Checkbox("SMAA",&params.useSMAA)){
        if(params.useSMAA){
            smaa = std::make_shared<SMAA>(renderWidth, renderHeight);
            smaa->loadShader(params.smaaQuality);
        }else{
            smaa.reset();
        }
    }
    if(smaa){
        smaa->renderImGui();
    }


    if(ImGui::Checkbox("SSAO",&params.useSSAO)){
        if(params.useSSAO){
            ssao = std::make_shared<SSAO>(renderWidth, renderHeight);
        }else{
            ssao.reset();
        }
        lighting.ssaoTexture = ssao ? ssao->bluredTexture : blackDummyTexture;
        ddo.setDeferredFramebuffer(&gbuffer,ssao ? ssao->bluredTexture : blackDummyTexture);
    }
    if(ssao){
        ssao->renderImGui();
    }


    ImGui::Checkbox("showLightingImgui",&showLightingImgui);

    ImGui::End();

    if(showLightingImgui){
        lighting.renderImGui(&showLightingImgui);
    }
}

}
