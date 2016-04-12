#include "saiga/rendering/deferred_renderer.h"
#include "saiga/util/error.h"
#include "saiga/geometry/triangle_mesh_generator.h"
#include "saiga/opengl/shader/shaderLoader.h"

#include "saiga/camera/camera.h"
#include "saiga/rendering/renderer.h"


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





Deferred_Renderer::Deferred_Renderer(int w, int h, RenderingParameters params):params(params),lighting(deferred_framebuffer) {
    setSize(w,h);
    lighting.init(w,h);

    deferred_framebuffer.init(w,h,params.gbp);

    ssao_framebuffer.create();
    Texture* ssaotex = new Texture();
    ssaotex->createEmptyTexture(w,h,GL_RED,GL_R8,GL_UNSIGNED_BYTE);
    ssao_framebuffer.attachTexture( std::shared_ptr<raw_Texture>(ssaotex) );
    ssao_framebuffer.drawToAll();
    ssao_framebuffer.check();

    glClearColor(1.0f,1.0f,1.0f,1.0f);
    glClear( GL_COLOR_BUFFER_BIT );
    glClearColor(0.0f,0.0f,0.0f,0.0f);

    lighting.ssaoTexture = ssaotex;
    ssao_framebuffer.unbind();

    postProcessor.init(w, h,&deferred_framebuffer,params.ppp,lighting.lightAccumulationTexture);


    auto qb = TriangleMeshGenerator::createFullScreenQuadMesh();
    qb->createBuffers(quadMesh);

    timers.resize(DeferredTimings::COUNT);
    for(auto &t : timers){
        t.create();
    }

    ssaoShader  =  ShaderLoader::instance()->load<SSAOShader>("ssao.glsl");
    blitDepthShader = ShaderLoader::instance()->load<MVPTextureShader>("blitDepth.glsl");

}

Deferred_Renderer::~Deferred_Renderer()
{

}



void Deferred_Renderer::resize(int width, int height)
{
    if (width <= 0 || height <= 0){
        cout << "Warning: The framebuffer size must be greater than zero to be complete." << endl;
        width = glm::max(width, 1);
        height = glm::max(height, 1);
    }
    cout << "Resizing Gbuffer to : " << width << "," << height << endl;
    setSize(width,height);
    postProcessor.resize(width,height);
    deferred_framebuffer.resize(width,height);
    ssao_framebuffer.resize(width,height);
    clearSSAO();
    lighting.resize(width,height);
}

void Deferred_Renderer::clearSSAO()
{
    ssao_framebuffer.bind();
    //clear with 1 -> no ambient occlusion
    glClearColor(1.0f,1.0f,1.0f,1.0f);
    glClear( GL_COLOR_BUFFER_BIT );
    glClearColor(0.0f,0.0f,0.0f,0.0f);
    ssao_framebuffer.unbind();
}




void Deferred_Renderer::toggleSSAO()
{
    clearSSAO();
    ssao = !ssao;
}

void Deferred_Renderer::render_intern(){

    startTimer(TOTAL);

    // When GL_FRAMEBUFFER_SRGB is disabled, the system assumes that the color written by the fragment shader
    // is in whatever colorspace the image it is being written to is. Therefore, no colorspace correction is performed.
    // If GL_FRAMEBUFFER_SRGB is enabled however, then if the destination image is in the sRGB colorspace
    // (as queried through glGetFramebufferAttachmentParameter(GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING)​),
    // then it will assume the shader's output is in the linear RGB colorspace.
    // It will therefore convert the output from linear RGB to sRGB.
    if(params.srgbWrites)
        glEnable(GL_FRAMEBUFFER_SRGB); //no reason to switch it off

    (*currentCamera)->recalculatePlanes();

    renderGBuffer(*currentCamera);
    assert_no_glerror();

    renderSSAO(*currentCamera);


    lighting.cullLights(*currentCamera);
    renderDepthMaps();


    //    glDisable(GL_DEPTH_TEST);
    //    glViewport(0,0,width,height);


    //copy depth to lighting framebuffer. that is needed for stencil culling




    //    mix_framebuffer.bind();
    //    glClear( GL_COLOR_BUFFER_BIT );


    renderLighting(*currentCamera);

//    startTimer(LIGHTACCUMULATION);
//    postProcessor.nextFrame();
//    postProcessor.bindCurrentBuffer();

//    lighting.renderLightAccumulation();
//    stopTimer(LIGHTACCUMULATION);

    if(params.writeDepthToOverlayBuffer){
//        writeGbufferDepthToCurrentFramebuffer();
    }else{
        glClear(GL_DEPTH_BUFFER_BIT);
    }

    startTimer(OVERLAY);
    renderer->renderOverlay(*currentCamera);
    stopTimer(OVERLAY);


    //write depth to default framebuffer
    if(params.writeDepthToDefaultFramebuffer){
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        writeGbufferDepthToCurrentFramebuffer();
    }

    postProcessor.nextFrame();
    postProcessor.bindCurrentBuffer();
//    postProcessor.switchBuffer();


    startTimer(POSTPROCESSING);
    postProcessor.render();
    stopTimer(POSTPROCESSING);

    //    deferred_framebuffer.blitDepth(0);




    startTimer(FINAL);
    renderer->renderFinal(*currentCamera);
    stopTimer(FINAL);


    stopTimer(TOTAL);

    //    std::cout<<"Time spent on the GPU: "<< getTime(TOTAL) <<std::endl;

    //    printTimings();


//    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);   // Make sure no FBO is set as the draw framebuffer
//     glBindFramebuffer(GL_READ_FRAMEBUFFER, lighting.lightAccumulationBuffer.getId()); // Make sure your multisampled FBO is the read framebuffer
//     glDrawBuffer(GL_BACK);                       // Set the back buffer as the draw buffer
//     glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST);

    assert_no_glerror();

}

void Deferred_Renderer::renderGBuffer(Camera *cam){
    startTimer(GEOMETRYPASS);

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);


    deferred_framebuffer.bind();
    glViewport(0,0,width,height);
    glClearColor(clearColor.x,clearColor.y,clearColor.z,clearColor.w);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);


    if(offsetGeometry){
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(offsetFactor,offsetUnits);
    }

    if(wireframe){
        glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
        glLineWidth(wireframeLineSize);
    }
    renderer->render(cam);
    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );


    if(offsetGeometry){
        glDisable(GL_POLYGON_OFFSET_FILL);
    }

    deferred_framebuffer.unbind();


    stopTimer(GEOMETRYPASS);

    assert_no_glerror();

}

void Deferred_Renderer::renderDepthMaps(){

    startTimer(DEPTHMAPS);

    // When GL_POLYGON_OFFSET_FILL, GL_POLYGON_OFFSET_LINE, or GL_POLYGON_OFFSET_POINT is enabled,
    // each fragment's depth value will be offset after it is interpolated from the depth values of the appropriate vertices.
    // The value of the offset is factor×DZ+r×units, where DZ is a measurement of the change in depth relative to the screen area of the polygon,
    // and r is the smallest value that is guaranteed to produce a resolvable offset for a given implementation.
    // The offset is added before the depth test is performed and before the value is written into the depth buffer.
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(2.0f,4.0f);
    lighting.renderDepthMaps(renderer);
    glDisable(GL_POLYGON_OFFSET_FILL);

    stopTimer(DEPTHMAPS);

    assert_no_glerror();

}

void Deferred_Renderer::renderLighting(Camera *cam){
    startTimer(LIGHTING);

    mat4 model;
    cam->getModelMatrix(model);
    lighting.setViewProj(model,cam->view,cam->proj);
    lighting.render(cam);

    stopTimer(LIGHTING);

    assert_no_glerror();
}

void Deferred_Renderer::renderSSAO(Camera *cam)
{

    startTimer(SSAO);
    if(ssao){

        ssao_framebuffer.bind();


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

    stopTimer(SSAO);

    assert_no_glerror();

}

void Deferred_Renderer::writeGbufferDepthToCurrentFramebuffer()
{
    glColorMask(GL_FALSE,GL_FALSE,GL_FALSE,GL_FALSE);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDepthFunc(GL_ALWAYS);
    blitDepthShader->bind();
    blitDepthShader->uploadTexture(deferred_framebuffer.getTextureDepth().get());
    quadMesh.bindAndDraw();
    blitDepthShader->unbind();
    glDepthFunc(GL_LESS);
    glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);

    assert_no_glerror();
}

void Deferred_Renderer::printTimings()
{
    cout<<"===================================="<<endl;
    cout<<"Geometry pass: "<<getTime(GEOMETRYPASS)<<"ms"<<endl;
    cout<<"SSAO: "<<getTime(SSAO)<<"ms"<<endl;
    cout<<"Depthmaps: "<<getTime(DEPTHMAPS)<<"ms"<<endl;
    cout<<"Lighting: "<<getTime(LIGHTING)<<"ms"<<endl;
    lighting.printTimings();
//    cout<<"Light accumulation: "<<getTime(LIGHTACCUMULATION)<<"ms"<<endl;
    cout<<"Overlay pass: "<<getTime(OVERLAY)<<"ms"<<endl;
    cout<<"Postprocessing: "<<getTime(POSTPROCESSING)<<"ms"<<endl;
    postProcessor.printTimings();
    cout<<"Final pass: "<<getTime(FINAL)<<"ms"<<endl;
    float total = getTime(TOTAL);
    cout<<"Total: "<<total<<"ms ("<< 1000/total << " fps)"<< endl;
    cout<<"===================================="<<endl;

}

