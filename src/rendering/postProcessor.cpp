#include "saiga/rendering/postProcessor.h"
#include "saiga/geometry/triangle_mesh_generator.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/util/error.h"

void PostProcessingShader::checkUniforms(){
    Shader::checkUniforms();
    location_texture = Shader::getUniformLocation("image");
    location_screenSize = Shader::getUniformLocation("screenSize");
    location_gbufferDepth = Shader::getUniformLocation("gbufferDepth");
    location_gbufferNormals = Shader::getUniformLocation("gbufferNormals");
    location_gbufferColor = Shader::getUniformLocation("gbufferColor");
    location_gbufferData = Shader::getUniformLocation("gbufferData");
}


void PostProcessingShader::uploadTexture(raw_Texture *texture){
    texture->bind(0);
    Shader::upload(location_texture,0);
}

void PostProcessingShader::uploadGbufferTextures(GBuffer *gbuffer)
{
    gbuffer->getTextureDepth()->bind(1);
    Shader::upload(location_gbufferDepth,1);
    gbuffer->getTextureNormal()->bind(2);
    Shader::upload(location_gbufferNormals,2);
    gbuffer->getTextureColor()->bind(3);
    Shader::upload(location_gbufferColor, 3);
    gbuffer->getTextureData()->bind(4);
    Shader::upload(location_gbufferData, 4);
}

void PostProcessingShader::uploadScreenSize(vec4 size){
    Shader::upload(location_screenSize,size);
}



void BrightnessShader::checkUniforms()
{
    PostProcessingShader::checkUniforms();
    location_brightness = Shader::getUniformLocation("brightness");
}

void BrightnessShader::uploadAdditionalUniforms()
{
    Shader::upload(location_brightness, brightness);
}



void LightAccumulationShader::checkUniforms(){
    DeferredShader::checkUniforms();
    location_lightAccumulationtexture = Shader::getUniformLocation("lightAccumulationtexture");
}


void LightAccumulationShader::uploadLightAccumulationtexture(raw_Texture *texture){
    texture->bind(4);
    Shader::upload(location_lightAccumulationtexture,4);
}




void PostProcessor::init(int width, int height, GBuffer* gbuffer, PostProcessorParameters params, Texture *LightAccumulationTexture)
{
    this->params = params;
    this->width=width;this->height=height;
    this->gbuffer = gbuffer;
//    this->gbufferDepth = gbuffer->getTextureDepth();
//    this->gbufferNormals = gbuffer->getTextureNormal();
//    this->gbufferColor = gbuffer->getTextureColor();
    createFramebuffers();

    auto qb = TriangleMeshGenerator::createFullScreenQuadMesh();
    qb->createBuffers(quadMesh);


    this->LightAccumulationTexture = LightAccumulationTexture;

    //    computeTest = ShaderLoader::instance()->load<Shader>("computeTest.glsl");
    assert_no_glerror();
}

void PostProcessor::nextFrame()
{
//    gbuffer->blitDepth(framebuffers[0].getId());
    currentBuffer = 0;
    lastBuffer = 1;

}

void PostProcessor::createFramebuffers()
{
    for(int i = 0 ;i <2 ;++i){
        framebuffers[i].create();

        if(i==0){
//            Texture* depth_stencil = new Texture();
//            depth_stencil->createEmptyTexture(width,height,GL_DEPTH_STENCIL, GL_DEPTH24_STENCIL8,GL_UNSIGNED_INT_24_8);
//            framebuffers[i].attachTextureDepthStencil(depth_stencil);
           Texture* depth = new Texture();
           depth->createEmptyTexture(width,height,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT32,GL_UNSIGNED_SHORT);
           framebuffers[i].attachTextureDepth( framebuffer_texture_t(depth) );
        }


        textures[i] = new Texture();
        if(params.srgb){
            textures[i]->createEmptyTexture(width,height,GL_RGBA,GL_SRGB8_ALPHA8,GL_UNSIGNED_BYTE);
        }else{
            switch(params.quality){
            case Quality::LOW:
                textures[i]->createEmptyTexture(width,height,GL_RGBA,GL_RGBA8,GL_UNSIGNED_BYTE);
                break;
            case Quality::MEDIUM:
                textures[i]->createEmptyTexture(width,height,GL_RGBA,GL_RGBA16,GL_UNSIGNED_SHORT);
                break;
            case Quality::HIGH:
                textures[i]->createEmptyTexture(width,height,GL_RGBA,GL_RGBA16,GL_UNSIGNED_SHORT);
                break;
            }
        }
        framebuffers[i].attachTexture( framebuffer_texture_t(textures[i]) );
        framebuffers[i].drawToAll();
        framebuffers[i].check();
        framebuffers[i].unbind();
    }
}

void PostProcessor::bindCurrentBuffer()
{
    framebuffers[currentBuffer].bind();
}

void PostProcessor::switchBuffer()
{
    lastBuffer = currentBuffer;
    currentBuffer = (currentBuffer +1) %2;
}

void PostProcessor::render()
{
    int effects = postProcessingEffects.size();

    if(effects==0){
        cout<<"Warning no post processing effects specified. The screen will probably be black!"<<endl;
        return;
    }


    first = true;


    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);

    for(int i = 0 ; i < effects; ++i){
        switchBuffer();
        shaderTimer[i].startTimer();
        applyShader(postProcessingEffects[i]);
        shaderTimer[i].stopTimer();
    }

//    shaderTimer[effects-1].startTimer();
//    applyShaderFinal(postProcessingEffects[effects-1]);
//    shaderTimer[effects-1].stopTimer();

    glEnable(GL_DEPTH_TEST);



    //    std::cout<<"Time spent on the GPU: "<< timer.getTimeMS() <<std::endl;
}

void PostProcessor::setPostProcessingEffects(const std::vector<PostProcessingShader *> &postProcessingEffects){
    assert_no_glerror();
    this->postProcessingEffects = postProcessingEffects;
    shaderTimer.clear();
    shaderTimer.resize(postProcessingEffects.size());
    for(auto &t : shaderTimer){
        t.create();
    }
    assert_no_glerror();
}

void PostProcessor::printTimings()
{
    for(unsigned int i = 0 ; i < postProcessingEffects.size() ; ++i){
        cout<<"\t"<<shaderTimer[i].getTimeMS()<<"ms "<<postProcessingEffects[i]->name<<endl;
    }
}

void PostProcessor::resize(int width, int height)
{
    this->width=width;this->height=height;
    framebuffers[0].resize(width,height);
    framebuffers[1].resize(width,height);
}

void PostProcessor::applyShader(PostProcessingShader *postProcessingShader)
{
    framebuffers[currentBuffer].bind();


    postProcessingShader->bind();
    vec4 screenSize(width,height,1.0/width,1.0/height);
    postProcessingShader->uploadScreenSize(screenSize);
    postProcessingShader->uploadTexture( (first) ? LightAccumulationTexture : textures[lastBuffer] );
    postProcessingShader->uploadGbufferTextures(gbuffer);
    postProcessingShader->uploadAdditionalUniforms();
    quadMesh.bindAndDraw();
    postProcessingShader->unbind();

//    framebuffers[currentBuffer].unbind();

    first = false;

}

void PostProcessor::blitLast(int windowWidth, int windowHeight){
//    framebuffers[lastBuffer].blitColor(0);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffers[currentBuffer].getId());
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glBlitFramebuffer(0, 0, width, height, 0, 0, windowWidth, windowHeight,GL_COLOR_BUFFER_BIT, GL_LINEAR);

}

void PostProcessor::applyShaderFinal(PostProcessingShader *postProcessingShader)
{
    //compute shader test

    //    computeTest->bind();
    //    textures[lastBuffer]->bindImageTexture(3,GL_WRITE_ONLY);
    //    glm::uvec3 problemSize(1000,500,1);
    //    auto groups = computeTest->getNumGroupsCeil(problemSize);
    //    computeTest->dispatchCompute(groups);
    //    computeTest->memoryBarrierTextureFetch();
    //    computeTest->unbind();
    //    Error::quitWhenError("compute shader stuff");



//    glEnable(GL_FRAMEBUFFER_SRGB);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);


    postProcessingShader->bind();
    vec4 screenSize(width,height,1.0/width,1.0/height);
    postProcessingShader->uploadScreenSize(screenSize);
    postProcessingShader->uploadTexture((first) ? LightAccumulationTexture : textures[lastBuffer]);
    postProcessingShader->uploadGbufferTextures(gbuffer);
    postProcessingShader->uploadAdditionalUniforms();
    quadMesh.bindAndDraw();
    postProcessingShader->unbind();

    first = false;
    //    glDisable(GL_FRAMEBUFFER_SRGB);
}


