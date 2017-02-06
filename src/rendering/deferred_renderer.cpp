#include "saiga/rendering/deferred_renderer.h"
#include "saiga/util/error.h"
#include "saiga/geometry/triangle_mesh_generator.h"
#include "saiga/opengl/shader/shaderLoader.h"

#include "saiga/camera/camera.h"
#include "saiga/rendering/renderer.h"



Deferred_Renderer::Deferred_Renderer(int windowWidth, int windowHeight, RenderingParameters params) :
    windowWidth(windowWidth), windowHeight(windowHeight),
    width(windowWidth*params.renderScale), height(windowHeight*params.renderScale),
    params(params), lighting(gbuffer)
{
    cameraBuffer.createGLBuffer(nullptr,sizeof(CameraDataGLSL),GL_DYNAMIC_DRAW);

    //    setSize(windowWidth,windowHeight);

    if(params.useSMAA)
        smaa.init(windowWidth*params.renderScale, windowHeight*params.renderScale,params.smaaQuality);
    else
        smaa.init(2,2,SMAA::Quality::SMAA_PRESET_LOW);

    if(params.useSSAO)
        ssao.init(windowWidth*params.renderScale, windowHeight*params.renderScale);
    else
        ssao.init(2,2);

    if(params.srgbWrites){

        //intel graphics drivers on windows do not define this extension but srgb still works..
        //SAIGA_ASSERT(hasExtension("GL_EXT_framebuffer_sRGB"));

        //Mesa drivers do not respect the spec when blitting with srgb framebuffers.
        //https://lists.freedesktop.org/archives/mesa-dev/2015-February/077681.html

        //TODO check for mesa
        //If this is true some recording softwares record the image too dark :(
        blitLastFramebuffer = false;
    }



    gbuffer.init(width, height, params.gbp);

    lighting.init(width, height, params.useGPUTimers);
    lighting.shadowSamples = params.shadowSamples;
    lighting.loadShaders();

    lighting.ssaoTexture = ssao.bluredTexture;

    postProcessor.init(width, height, &gbuffer, params.ppp, lighting.lightAccumulationTexture, params.useGPUTimers);


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

    cout << "Deferred Renderer initialized. Render resolution: " << width << "x" << height << endl;

}

Deferred_Renderer::~Deferred_Renderer()
{

}



void Deferred_Renderer::resize(int windowWidth, int windowHeight)
{


    if (windowWidth <= 0 || windowHeight <= 0) {
        cout << "Warning: The window size must be greater than zero to be complete." << endl;
        windowWidth = glm::max(windowWidth, 1);
        windowHeight = glm::max(windowHeight, 1);
    }
    this->windowWidth = windowWidth;
    this->windowHeight = windowHeight;
    this->width = windowWidth * params.renderScale;
    this->height = windowHeight * params.renderScale;
    cout << "Resizing Window to : " << windowWidth << "," << windowHeight << endl;
    cout << "Framebuffer size: " << width << " " << height << endl;
    postProcessor.resize(width, height);
    gbuffer.resize(width, height);
    lighting.resize(width, height);

    if(params.useSSAO)
        ssao.resize(width, height);

    if(params.useSMAA)
        smaa.resize(width,height);
}






void Deferred_Renderer::render_intern() {

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

    (*currentCamera)->recalculatePlanes();
    bindCamera(*currentCamera);
    renderGBuffer(*currentCamera);
    assert_no_glerror();


    renderSSAO(*currentCamera);
    //    return;

    lighting.cullLights(*currentCamera);
    renderDepthMaps();


    //    glDisable(GL_DEPTH_TEST);
    //    glViewport(0,0,width,height);


    //copy depth to lighting framebuffer. that is needed for stencil culling




    //    mix_framebuffer.bind();
    //    glClear( GL_COLOR_BUFFER_BIT );


    bindCamera(*currentCamera);
    renderLighting(*currentCamera);


    //    startTimer(LIGHTACCUMULATION);
    //    postProcessor.nextFrame();
    //    postProcessor.bindCurrentBuffer();

    //    lighting.renderLightAccumulation();
    //    stopTimer(LIGHTACCUMULATION);

    if (params.writeDepthToOverlayBuffer) {
        //        writeGbufferDepthToCurrentFramebuffer();
    }
    else {
        glClear(GL_DEPTH_BUFFER_BIT);
    }

    startTimer(OVERLAY);

    bindCamera(*currentCamera);
    renderer->renderOverlay(*currentCamera);
    stopTimer(OVERLAY);



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
        smaa.render(postProcessor.getCurrentTexture(),postProcessor.getTargetBuffer());
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
    //glClear(GL_COLOR_BUFFER_BIT);
    startTimer(FINAL);
    renderer->renderFinal(*currentCamera);
    stopTimer(FINAL);

    glDisable(GL_BLEND);

    if(blitLastFramebuffer)
        postProcessor.blitLast(windowWidth, windowHeight);
    else
        postProcessor.renderLast(windowWidth, windowHeight);

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
    glViewport(0, 0, width, height);
    glClearColor(clearColor.x, clearColor.y, clearColor.z, clearColor.w);

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



    if (offsetGeometry) {
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(offsetFactor, offsetUnits);
    }

    if (wireframe) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glLineWidth(wireframeLineSize);
    }
    renderer->render(cam);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);


    if (offsetGeometry) {
        glDisable(GL_POLYGON_OFFSET_FILL);
    }

    glDisable(GL_STENCIL_TEST);

    gbuffer.unbind();


    stopTimer(GEOMETRYPASS);

    assert_no_glerror();

}

void Deferred_Renderer::renderDepthMaps() {

    startTimer(DEPTHMAPS);

    // When GL_POLYGON_OFFSET_FILL, GL_POLYGON_OFFSET_LINE, or GL_POLYGON_OFFSET_POINT is enabled,
    // each fragment's depth value will be offset after it is interpolated from the depth values of the appropriate vertices.
    // The value of the offset is factor×DZ+r×units, where DZ is a measurement of the change in depth relative to the screen area of the polygon,
    // and r is the smallest value that is guaranteed to produce a resolvable offset for a given implementation.
    // The offset is added before the depth test is performed and before the value is written into the depth buffer.
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(2.0f, 4.0f);
    lighting.renderDepthMaps(renderer);
    glDisable(GL_POLYGON_OFFSET_FILL);

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
        ssao.render(cam, &gbuffer);


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
    blitDepthShader->uploadTexture(gbuffer.getTextureDepth().get());
    quadMesh.bindAndDraw();
    blitDepthShader->unbind();
    glDepthFunc(GL_LESS);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

    assert_no_glerror();
}

void Deferred_Renderer::bindCamera(Camera *cam)
{
    CameraDataGLSL cd(cam);
    cameraBuffer.updateBuffer(&cd,sizeof(CameraDataGLSL),0);
    cameraBuffer.bind(CAMERA_DATA_BINDING_POINT);
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

