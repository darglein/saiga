/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/rendering/lighting/ssao.h"

#include "saiga/core/image/imageGenerator.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/math/random.h"
#include "saiga/core/model/model_from_shape.h"
#include "saiga/opengl/shader/shaderLoader.h"

namespace Saiga
{
void SSAOShader::checkUniforms()
{
    DeferredShader::checkUniforms();
    location_invProj       = getUniformLocation("invProj");
    location_randomImage   = getUniformLocation("randomImage");
    location_kernelSize    = getUniformLocation("uKernelSize");
    location_kernelOffsets = getUniformLocation("uKernelOffsets");
    location_radius        = getUniformLocation("radius");
    location_power         = getUniformLocation("power");
}



void SSAOShader::uploadInvProj(mat4& mat)
{
    Shader::upload(location_invProj, mat);
}

void SSAOShader::uploadData()
{
    Shader::upload(location_kernelSize, (int)kernelOffsets.size());
    Shader::upload(location_kernelOffsets, kernelOffsets.size(), kernelOffsets.data());

    Shader::upload(location_radius, radius);
    Shader::upload(location_power, exponent);
}

void SSAOShader::uploadRandomImage(std::shared_ptr<Texture> img)
{
    Shader::upload(location_randomImage, img, 4);
}


SSAO::SSAO(int w, int h) : quadMesh(FullScreenQuad())
{
    init(w, h);
}

void SSAO::init(int w, int h)
{
    screenSize = vec2(w, h);
    ssaoSize   = ivec2(w / 2, h / 2);

    ssao_framebuffer.create();
    ssaotex = std::make_shared<Texture>();
    ssaotex->create(ssaoSize[0], ssaoSize[1], GL_RED, GL_R8, GL_UNSIGNED_BYTE);
    ssao_framebuffer.attachTexture(ssaotex);
    ssao_framebuffer.drawToAll();
    ssao_framebuffer.check();
    ssao_framebuffer.unbind();

    ssao_framebuffer2.create();
    bluredTexture = std::make_shared<Texture>();
    bluredTexture->create(ssaoSize[0], ssaoSize[1], GL_RED, GL_R8, GL_UNSIGNED_BYTE);
    ssao_framebuffer2.attachTexture(bluredTexture);
    ssao_framebuffer2.drawToAll();
    ssao_framebuffer2.check();
    ssao_framebuffer2.unbind();


    ssaoShader = shaderLoader.load<SSAOShader>("post_processing/ssao2.glsl");
    blurShader = shaderLoader.load<MVPTextureShader>("post_processing/ssao_blur.glsl");

    setKernelSize(kernelSize);


    clearSSAO();


    assert_no_glerror();

    std::cout << "SSAO initialized!" << std::endl;
}

void SSAO::resize(int w, int h)
{
    screenSize  = vec2(w, h);
    ssaoSize    = ivec2(w / 2, h / 2);
    ssaoSize[0] = std::max(ssaoSize[0], 1);
    ssaoSize[1] = std::max(ssaoSize[1], 1);

    ssao_framebuffer.resize(ssaoSize[0], ssaoSize[1]);
    ssao_framebuffer2.resize(ssaoSize[0], ssaoSize[1]);
    clearSSAO();
}

void SSAO::clearSSAO()
{
    ssao_framebuffer2.bind();
    // clear with 1 -> no ambient occlusion
    //    glClearColor(1.0f,1.0f,1.0f,1.0f);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    //    glClearColor(0.0f,0.0f,0.0f,0.0f);
    ssao_framebuffer2.unbind();
}

void SSAO::render(Camera* cam, const ViewPort& vp, GBuffer* gbuffer)
{
    glViewport(0, 0, ssaoSize[0], ssaoSize[1]);
    ssao_framebuffer.bind();


    if(ssaoShader->bind())
    {
        //    gbuffer->clampToEdge();
        ssaoShader->uploadScreenSize(vp.getVec4());
        ssaoShader->uploadFramebuffer(gbuffer);
        ssaoShader->uploadRandomImage(randomTexture);
        ssaoShader->uploadData();
        mat4 iproj = inverse(cam->proj);
        ssaoShader->uploadInvProj(iproj);
        quadMesh.BindAndDraw();
        ssaoShader->unbind();
    }

    ssao_framebuffer.unbind();



    ssao_framebuffer2.bind();

    if(blurShader->bind())
    {
        blurShader->uploadTexture(ssaotex.get());
        quadMesh.BindAndDraw();
        blurShader->unbind();
    }

    ssao_framebuffer2.unbind();


    glViewport(0, 0, screenSize[0], screenSize[1]);
}


void SSAO::setKernelSize(int _kernelSize)
{
    kernelSize = _kernelSize;
    kernelOffsets.resize(kernelSize);
    for (int i = 0; i < kernelSize; ++i)
    {
        vec3 sample = normalize(linearRand(vec3(-1, -1, 0), vec3(1, 1, 1)));
        float scale = float(i) / float(kernelSize);
        scale       = mix(0.1f, 1.0f, scale * scale);
        sample *= scale;

        //        vec3 sample = ballRand(1.0f);
        //        sample[2] = abs(sample[2]);

        kernelOffsets[i] = sample;
    }
    ssaoShader->kernelOffsets = kernelOffsets;

    auto randomImage = ImageGenerator::randomNormalized(32, 32);
    randomTexture    = std::make_shared<Texture>(*randomImage);
    randomTexture->setWrap(GL_REPEAT);
}

void SSAO::renderImGui()
{
    ImGui::PushID("SSAO::renderImGui");
    ImGui::InputInt("kernelSize", &kernelSize, 1, 8);
    ImGui::InputFloat("radius", &ssaoShader->radius, 1);
    ImGui::SliderFloat("exponent", &ssaoShader->exponent, 0, 1);
    if (ImGui::Button("Reload"))
    {
        setKernelSize(kernelSize);
    }
    ImGui::PopID();
}

}  // namespace Saiga
