#include "saiga/rendering/deferred_renderer.h"
#include "saiga/util/error.h"
#include "saiga/geometry/triangle_mesh_generator.h"
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





Deferred_Renderer::Deferred_Renderer():lighting(deferred_framebuffer){

}

Deferred_Renderer::~Deferred_Renderer()
{

}

void Deferred_Renderer::init(int w, int h){
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




    auto qb = TriangleMeshGenerator::createFullScreenQuadMesh();
    qb->createBuffers(quadMesh);

    timers.resize(DeferredTimings::COUNT);
    for(auto &t : timers){
        t.create();
    }
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
    lighting.resize(width,height);
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

    startTimer(TOTAL);

    (*currentCamera)->recalculatePlanes();

    renderGBuffer(*currentCamera);


    renderSSAO(*currentCamera);



    lighting.cullLights(*currentCamera);
    renderDepthMaps();



    glDisable(GL_DEPTH_TEST);
    glViewport(0,0,width,height);

    Error::quitWhenError("Deferred_Renderer::before blit");

    //copy depth to lighting framebuffer. that is needed for stencil culling


    Error::quitWhenError("Deferred_Renderer::after blit");


    //    mix_framebuffer.bind();
    //    glClear( GL_COLOR_BUFFER_BIT );


    renderLighting(*currentCamera);


    startTimer(LIGHTACCUMULATION);
    postProcessor.nextFrame(&deferred_framebuffer);
    postProcessor.bindCurrentBuffer();
    lighting.renderLightAccumulation();
    stopTimer(LIGHTACCUMULATION);

    startTimer(OVERLAY);
    renderer->renderOverlay(*currentCamera);
    stopTimer(OVERLAY);

    postProcessor.switchBuffer();


    startTimer(POSTPROCESSING);
    postProcessor.render();
    stopTimer(POSTPROCESSING);


    startTimer(FINAL);
    renderer->renderFinal(*currentCamera);
    stopTimer(FINAL);


    stopTimer(TOTAL);

    //    std::cout<<"Time spent on the GPU: "<< getTime(TOTAL) <<std::endl;

    //    printTimings();

    Error::quitWhenError("Deferred_Renderer::render_intern");

}

void Deferred_Renderer::renderGBuffer(Camera *cam){
    startTimer(GEOMETRYPASS);

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);


    deferred_framebuffer.bind();
    glViewport(0,0,width,height);
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

    Error::quitWhenError("Deferred_Renderer::renderGBuffer");

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

    Error::quitWhenError("Deferred_Renderer::renderDepthMaps");

}

void Deferred_Renderer::renderLighting(Camera *cam){
    startTimer(LIGHTING);

    mat4 model;
    cam->getModelMatrix(model);
    lighting.setViewProj(model,cam->view,cam->proj);
    lighting.render(cam);

    stopTimer(LIGHTING);

    Error::quitWhenError("Deferred_Renderer::renderLighting");
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

    Error::quitWhenError("Deferred_Renderer::renderSSAO");

}

void Deferred_Renderer::printTimings()
{
    cout<<"===================================="<<endl;
    cout<<"Geometry pass: "<<getTime(GEOMETRYPASS)<<"ms"<<endl;
    cout<<"SSAO: "<<getTime(SSAO)<<"ms"<<endl;
    cout<<"Depthmaps: "<<getTime(DEPTHMAPS)<<"ms"<<endl;
    cout<<"Lighting: "<<getTime(LIGHTING)<<"ms"<<endl;
    cout<<"Light accumulation: "<<getTime(LIGHTACCUMULATION)<<"ms"<<endl;
    cout<<"Overlay pass: "<<getTime(OVERLAY)<<"ms"<<endl;
    cout<<"Postprocessing: "<<getTime(POSTPROCESSING)<<"ms"<<endl;
    cout<<"Final pass: "<<getTime(FINAL)<<"ms"<<endl;
    cout<<"Total: "<<getTime(TOTAL)<<"ms"<<endl;
    cout<<"===================================="<<endl;

}

