#include "saiga/rendering/lighting/ssao.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/geometry/triangle_mesh_generator.h"
#include "saiga/opengl/texture/imageGenerator.h"
#include "saiga/rendering/gbuffer.h"

void SSAOShader::checkUniforms(){
    DeferredShader::checkUniforms();
    location_invProj = getUniformLocation("invProj");
    location_randomImage = getUniformLocation("randomImage");
    location_kernelSize = getUniformLocation("uKernelSize");
    location_kernelOffsets = getUniformLocation("uKernelOffsets");
    location_radius = getUniformLocation("radius");
    location_power = getUniformLocation("power");
}



void SSAOShader::uploadInvProj(mat4 &mat){
    Shader::upload(location_invProj,mat);
}

void SSAOShader::uploadData(){

    Shader::upload(location_kernelSize,(int)kernelOffsets.size());
    Shader::upload(location_kernelOffsets,kernelOffsets.size(),kernelOffsets.data());

    Shader::upload(location_radius,radius);
    Shader::upload(location_power,power);
}

void SSAOShader::uploadRandomImage(Texture *img)
{
    Shader::upload(location_randomImage,img,4);
}


SSAO::SSAO()
{

}

void SSAO::init(int w, int h)
{
    screenSize = vec2(w,h);
    ssaoSize = glm::ivec2(w/2,h/2);

    ssao_framebuffer.create();
    ssaotex = new Texture();
    ssaotex->createEmptyTexture(ssaoSize.x,ssaoSize.y,GL_RED,GL_R8,GL_UNSIGNED_BYTE);
    ssao_framebuffer.attachTexture( std::shared_ptr<raw_Texture>(ssaotex) );
    ssao_framebuffer.drawToAll();
    ssao_framebuffer.check();
    ssao_framebuffer.unbind();

    ssao_framebuffer2.create();
    bluredTexture = new Texture();
    bluredTexture->createEmptyTexture(ssaoSize.x,ssaoSize.y,GL_RED,GL_R8,GL_UNSIGNED_BYTE);
    ssao_framebuffer2.attachTexture( std::shared_ptr<raw_Texture>(bluredTexture) );
    ssao_framebuffer2.drawToAll();
    ssao_framebuffer2.check();
    ssao_framebuffer2.unbind();

    auto qb = TriangleMeshGenerator::createFullScreenQuadMesh();
    qb->createBuffers(quadMesh);

    ssaoShader  =  ShaderLoader::instance()->load<SSAOShader>("post_processing/ssao2.glsl");
    blurShader = ShaderLoader::instance()->load<MVPTextureShader>("post_processing/ssao_blur.glsl");

    setKernelSize(32);

    auto randomImage = ImageGenerator::randomNormalized(32,32);
    randomTexture = std::make_shared<Texture>();
    randomTexture->fromImage(*randomImage);
    randomTexture->setWrap(GL_REPEAT);

    clearSSAO();


    assert_no_glerror();

    cout << "SSAO initialized!" << endl;
}

void SSAO::resize(int w, int h)
{
    screenSize = vec2(w,h);
    ssaoSize = glm::ivec2(w/2,h/2);
	ssaoSize.x = glm::max(ssaoSize.x, 1);
	ssaoSize.y = glm::max(ssaoSize.y, 1);

    ssao_framebuffer.resize(ssaoSize.x,ssaoSize.y);
    ssao_framebuffer2.resize(ssaoSize.x,ssaoSize.y);
    clearSSAO();
}

void SSAO::clearSSAO()
{
    ssao_framebuffer2.bind();
    //clear with 1 -> no ambient occlusion
//    glClearColor(1.0f,1.0f,1.0f,1.0f);
    glClearColor(0.0f,0.0f,0.0f,0.0f);
    glClear( GL_COLOR_BUFFER_BIT );
//    glClearColor(0.0f,0.0f,0.0f,0.0f);
    ssao_framebuffer2.unbind();
}

void SSAO::render(Camera *cam, GBuffer* gbuffer)
{
    if(!ssao)
        return;

    glViewport(0,0,ssaoSize.x,ssaoSize.y);
    ssao_framebuffer.bind();


    ssaoShader->bind();

    //    gbuffer->clampToEdge();
    ssaoShader->uploadScreenSize(screenSize);
    ssaoShader->uploadFramebuffer(gbuffer);
    ssaoShader->uploadRandomImage(randomTexture.get());
    ssaoShader->uploadData();
    mat4 iproj = glm::inverse(cam->proj);
    ssaoShader->uploadInvProj(iproj);
    ssaoShader->bindCamera(cam);
    quadMesh.bindAndDraw();
    ssaoShader->unbind();

    ssao_framebuffer.unbind();




    ssao_framebuffer2.bind();

    blurShader->bind();

    blurShader->uploadTexture(ssaotex);
    quadMesh.bindAndDraw();
    blurShader->unbind();

    ssao_framebuffer2.unbind();


    glViewport(0,0,screenSize.x,screenSize.y);
}

void SSAO::setEnabled(bool enable)
{
    ssao = enable;
    clearSSAO();
}

void SSAO::toggle()
{
    setEnabled(!ssao);
}

void SSAO::setKernelSize(int kernelSize)
{

    kernelOffsets.resize(kernelSize);
    for (int i = 0; i < kernelSize; ++i) {
        vec3 sample = glm::normalize(glm::linearRand(vec3(-1,-1,0),vec3(1,1,1)));
        float scale = float(i) / float(kernelSize);
        scale = glm::mix(0.1f, 1.0f, scale * scale);
        sample *= scale;

//        vec3 sample = glm::ballRand(1.0f);
//        sample.z = glm::abs(sample.z);

        kernelOffsets[i] = sample;
    }
    ssaoShader->kernelOffsets = kernelOffsets;
}
